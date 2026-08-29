"""lmms-eval adapter for native Molmo2 and the V6 AdaCodec video path.

This adapter intentionally imports the production codec implementation from the
Molmo2 training repository instead of reimplementing its model or GOP contract.
The lmms-eval layer owns task/prompt plumbing; Molmo2 owns preprocessing,
timestamps, visual positions, generation, and codec feature construction.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

_VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def _strip_lmms_placeholders(text: str) -> str:
    """Remove task-side media placeholders; Molmo2 inserts its own tokens."""
    text = re.sub(r"<image(?:\s+\d+)?>", "", str(text))
    text = re.sub(r"<video(?:\s+\d+)?>", "", text)
    return text.strip()


def _normalize_visuals(value: Any) -> list[Any]:
    """Flatten the shallow lists returned by lmms-eval doc_to_visual hooks."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike, Image.Image, np.ndarray)):
        return [value]
    if isinstance(value, (list, tuple)):
        output: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                output.extend(item)
            else:
                output.append(item)
        return output
    return [value]


def _visual_kind(visuals: list[Any]) -> str:
    """Classify one lmms-eval sample without decoding its media."""
    if not visuals:
        return "text"
    path_visuals = [
        Path(os.fspath(value))
        for value in visuals
        if isinstance(value, (str, os.PathLike))
    ]
    if path_visuals and len(path_visuals) == len(visuals):
        if all(path.suffix.lower() in _VIDEO_SUFFIXES for path in path_visuals):
            if len(path_visuals) != 1:
                raise ValueError("Molmo2 currently supports one video per example")
            return "video"
        return "image"
    if all(isinstance(value, (Image.Image, np.ndarray)) for value in visuals):
        return "image"
    raise TypeError(
        "unsupported mixed visual payload; expected one video path or image(s), "
        f"got {[type(value).__name__ for value in visuals]}"
    )


def _ranked_trace_path(path: str, rank: int, world_size: int) -> str:
    """Give each data-parallel evaluator rank an independent JSONL file."""
    if world_size <= 1:
        return path
    return f"{path}.rank{rank}.jsonl"


def _select_foreign_visuals(
    all_visuals: list[list[Any]], target_index: int
) -> list[Any]:
    """Pick the next video donor whose physical path differs from target."""
    target = all_visuals[target_index]
    if _visual_kind(target) != "video":
        raise ValueError("foreign P evaluation requires video targets")
    target_path = os.path.realpath(os.fspath(target[0]))
    for offset in range(1, len(all_visuals)):
        candidate = all_visuals[(target_index + offset) % len(all_visuals)]
        if _visual_kind(candidate) != "video":
            continue
        if os.path.realpath(os.fspath(candidate[0])) != target_path:
            return candidate
    raise ValueError(
        "foreign P evaluation could not find a physically distinct donor video"
    )


def _codec_timeline_sampler_overrides(
    mode: str, *, max_frames: int, base_fps: float
) -> dict[str, object]:
    """Build the Molmo loader contract for one codec candidate timeline."""
    if mode == "prefix_2fps":
        return {
            "max_frames": int(max_frames),
            "frame_sample_mode": "fps",
            "candidate_sampling_fps": (float(base_fps),),
        }
    if mode in {"full_span_1fps", "full_span_2fps"}:
        sampling_fps = 1.0 if mode == "full_span_1fps" else float(base_fps)
        return {
            "max_frames": int(max_frames),
            "frame_sample_mode": "uniform_last_frame",
            # TimeSampler owns max_fps; FrameSampler owns min_fps. VideoLoader
            # applies only fields supported by its concrete sampler.
            "max_fps": sampling_fps,
            "min_fps": sampling_fps,
        }
    raise ValueError(
        "timeline_sampling_mode must be full_span_1fps, full_span_2fps, or "
        "prefix_2fps, "
        f"got {mode!r}"
    )


def _same_i_positions(roles: Any) -> tuple[int, ...]:
    """Return the exact I positions from an already selected codec timeline."""
    roles = tuple(roles)
    if not roles or roles[0] != "I":
        raise ValueError("same-I/no-P roles must start with an I-frame")
    if any(role not in {"I", "P"} for role in roles):
        raise ValueError("same-I/no-P roles must contain only I/P")
    return tuple(index for index, role in enumerate(roles) if role == "I")


class _Molmo2CodecRuntime:
    """Long-lived native Molmo2 runtime used by the lmms-eval adapter."""

    def __init__(
        self,
        *,
        training_repo: str,
        base_checkpoint: str | None,
        ptok_checkpoint: str,
        stage2_checkpoint: str | None,
        gamma_artifact: str | None,
        device: torch.device,
        video_backend: str,
        visual_token_budget: int,
        timeline_max_frames: int,
        timeline_sampling_mode: str,
        max_frames: int,
        seq_len: int,
        gop_cache_dir: str,
        gop_cache_read_dirs: str,
        motion_cache_dir: str,
        motion_cache_read_dirs: str,
        feature_cache_dir: str,
        p_variant: str,
    ) -> None:
        training_repo = os.path.abspath(os.path.expanduser(training_repo))
        if not os.path.isdir(os.path.join(training_repo, "codec")):
            raise FileNotFoundError(
                f"training_repo does not contain codec/: {training_repo}"
            )
        if training_repo not in sys.path:
            sys.path.insert(0, training_repo)

        # Import after adding the native repository. Keeping these imports local
        # lets ordinary lmms-eval model discovery work without Molmo2 installed.
        from codec.codec_cache import _atomic_save
        from codec.codec_config import ADACODEC, with_gamma_artifact
        from codec.codec_eval import (
            _pool_index_one_frame,
            eval_frozen_feature_cache_path,
            eval_motion_cache_path,
            eval_selected_gop_cache_path,
            evaluation_preprocessor_frame_counts,
            load_4b_cfg,
            load_eval_frozen_feature_cache,
            load_eval_motion_cache,
            load_eval_selected_gop,
            load_eval_selected_gop_artifact,
            replace_p_u5_with_foreign,
            resize_eval_rgb_frames,
            resolve_gamma_artifact,
            rewrite_tokenized_video_for_frame_counts,
            selected_eval_rgb_frames,
            store_eval_frozen_feature_cache,
        )
        from codec.codec_inference import CodecInferenceRuntime
        from codec.native_p_tokenizer import (
            NATIVE_P_TOKENIZATION_MODES,
            configured_p_tokenization,
            p_output_tokens,
        )
        from codec.codec_preprocess import assign_roles
        from codec.codec_representation import (
            HP,
            N_I,
            PATCH,
            codec_motion_roles,
            selected_u5_from_motion,
        )
        from codec.codec_runtime import codec_encode_u5
        from codec.gop_sampling import select_complete_gops
        from olmo.data.video_loader import VideoFrames
        from olmo.models.molmo2.molmo2 import Molmo2Config
        from olmo.torch_util import seed_all
        from olmo.train.checkpointer import load_model_state

        if video_backend not in {"codec", "dense"}:
            raise ValueError(
                f"video_backend must be 'codec' or 'dense', got {video_backend!r}"
            )
        if p_variant not in {
            "real",
            "drop_p",
            "zero_output",
            "shuffle_content",
            "foreign",
        }:
            raise ValueError(
                "p_variant must be real, drop_p, zero_output, shuffle_content, or "
                f"foreign; got {p_variant!r}"
            )
        self.training_repo = training_repo
        self.device = device
        # codec_motion reads this environment variable at call time. Each
        # accelerate worker must search on its own local GPU rather than all
        # workers contending for global cuda:0.
        os.environ["CODEC_MOTION_DEVICE"] = str(device)
        self.video_backend = video_backend
        self.p_variant = p_variant
        self.visual_token_budget = int(visual_token_budget)
        self.timeline_max_frames = int(timeline_max_frames)
        self.timeline_sampling_mode = str(timeline_sampling_mode)
        _codec_timeline_sampler_overrides(
            self.timeline_sampling_mode,
            max_frames=self.timeline_max_frames,
            base_fps=2.0,
        )
        self.max_frames = int(max_frames)
        self.seq_len = int(seq_len)
        self.p_tokenization = configured_p_tokenization()
        self.native_p_mode = (
            self.p_tokenization in NATIVE_P_TOKENIZATION_MODES
        )
        self._N_I = N_I
        self._VideoFrames = VideoFrames
        self._atomic_save = _atomic_save
        self._assign_roles = assign_roles
        self._codec_motion_roles = codec_motion_roles
        self._selected_u5_from_motion = selected_u5_from_motion
        self._select_complete_gops = select_complete_gops
        self._codec_encode_u5 = codec_encode_u5
        self._eval_motion_cache_path = eval_motion_cache_path
        self._eval_selected_gop_cache_path = eval_selected_gop_cache_path
        self._eval_frozen_feature_cache_path = eval_frozen_feature_cache_path
        self._load_eval_motion_cache = load_eval_motion_cache
        self._load_eval_selected_gop = load_eval_selected_gop
        self._load_eval_selected_gop_artifact = load_eval_selected_gop_artifact
        self._load_eval_frozen_feature_cache = load_eval_frozen_feature_cache
        self._replace_p_u5_with_foreign = replace_p_u5_with_foreign
        self._store_eval_frozen_feature_cache = store_eval_frozen_feature_cache
        self._resize_eval_rgb_frames = resize_eval_rgb_frames
        self._selected_eval_rgb_frames = selected_eval_rgb_frames
        self._rewrite_tokenized_video_for_frame_counts = (
            rewrite_tokenized_video_for_frame_counts
        )
        self._evaluation_preprocessor_frame_counts = (
            evaluation_preprocessor_frame_counts
        )

        def _cache_dirs(*values: str) -> list[str]:
            result: list[str] = []
            for value in values:
                for path in str(value or "").split(os.pathsep):
                    path = path.strip()
                    if path and path not in result:
                        result.append(path)
            return result

        # Prefix and full-span artifacts can have equal lengths while pointing
        # at different moments, so the sampling mode is a cache namespace.
        cache_namespace = f"timeline-{self.timeline_sampling_mode}"

        def _namespaced(path: str) -> str:
            return os.path.join(path, cache_namespace) if path else ""

        self.gop_cache_dir = _namespaced(gop_cache_dir)
        self.motion_cache_dir = _namespaced(motion_cache_dir)
        self.feature_cache_dir = _namespaced(feature_cache_dir)
        self.gop_cache_read_dirs = _cache_dirs(
            self.gop_cache_dir,
            *[
                _namespaced(path)
                for path in _cache_dirs(gop_cache_read_dirs)
            ],
        )
        self.motion_cache_read_dirs = _cache_dirs(
            self.motion_cache_dir,
            *[
                _namespaced(path)
                for path in _cache_dirs(motion_cache_read_dirs)
            ],
        )

        seed_all(6198)
        self.base_checkpoint, config = load_4b_cfg(base_checkpoint)
        if not isinstance(config, Molmo2Config):
            raise TypeError(f"expected Molmo2Config, got {type(config).__name__}")
        if config.mm_preprocessor.image is not None:
            config.mm_preprocessor.image.max_crops = 24
            config.mm_preprocessor.image.max_images = 20
        config.mm_preprocessor.video.max_frames = self.max_frames
        if "torchcodec" in (config.mm_preprocessor.video.loading_method or ""):
            config.mm_preprocessor.video.loading_method = "decord_with_av_fallback"
        config.llm.max_sequence_length = max(
            config.llm.max_sequence_length, self.seq_len
        )
        self.config = config

        gamma_artifact = resolve_gamma_artifact(gamma_artifact)
        self.codec_config = with_gamma_artifact(ADACODEC, gamma_artifact)
        self.ptok_checkpoint = (
            os.path.abspath(os.path.expanduser(ptok_checkpoint))
            if ptok_checkpoint
            else ""
        )
        self.stage2_checkpoint = (
            os.path.abspath(os.path.expanduser(stage2_checkpoint))
            if stage2_checkpoint
            else ""
        )

        # Native Pruning-16 / Compression-9 checkpoints contain a different
        # visual branch from the historical learned P-tokenizer.  Reusing the
        # feature-injection runtime below would silently load and evaluate the
        # wrong branch.  Build the exact training-side native model and patched
        # preprocessor instead; lmms-eval continues to own only task/prompt
        # plumbing.
        if self.native_p_mode:
            if self.p_variant != "real":
                raise ValueError(
                    "native P lmms-eval currently supports only p_variant=real"
                )
            if not self.stage2_checkpoint:
                raise ValueError(
                    "native P lmms-eval requires a Stage-2 checkpoint"
                )
            from codec.codec_pipeline import install_codec_preprocessor_patches
            from codec.native_p_eval import _load_native_model

            install_codec_preprocessor_patches()
            self.model, self.config = _load_native_model(
                self.base_checkpoint,
                self.stage2_checkpoint,
                self.device,
            )
            self.n_p_tokens = p_output_tokens(self.p_tokenization)
            self.tokenizer = self.config.build_tokenizer()
            self.image_patch_token_id = self.tokenizer.image_patch_token_id
            self.preprocessor = self.config.build_preprocessor(
                for_inference=True,
                is_training=False,
                text_seq_len=None,
                max_seq_len=self.seq_len,
                include_image=False,
            )
            self.preprocessor.video_preprocessor.video_backend = (
                "codec" if self.video_backend == "codec" else "native"
            )
            self._captured = {}
            self._inject = {"features": None}
            self.codec_runtime = None
            eval_logger.info(
                "Using training-identical native P path: {} / {}",
                self.p_tokenization,
                self.video_backend,
            )
            return

        eval_logger.info("Building native Molmo2 and loading 4B weights")
        if not self.ptok_checkpoint:
            raise ValueError(
                "ptok_checkpoint is required for the historical learned "
                "P-tokenizer path"
            )
        with torch.device("meta"):
            model = config.build_model()
        model.to_empty(device=device)
        load_model_state(self.base_checkpoint, model)
        self.model = model.to(device).eval()

        pool_index = torch.from_numpy(_pool_index_one_frame(HP, HP)).to(
            device=device, dtype=torch.long
        )
        self.n_p_tokens = self.codec_config.n_p_tokens(N_I)
        self.codec_runtime = CodecInferenceRuntime.from_checkpoints(
            self.model,
            pool_index,
            device=device,
            patch_size=PATCH,
            n_p_tokens=self.n_p_tokens,
            stage1_path=self.ptok_checkpoint,
            stage2_path=(self.stage2_checkpoint or None),
        )

        self.tokenizer = config.build_tokenizer()
        self.image_patch_token_id = self.tokenizer.image_patch_token_id
        self.preprocessor = config.build_preprocessor(
            for_inference=True,
            is_training=False,
            text_seq_len=None,
            max_seq_len=self.seq_len,
            include_image=False,
        )
        self._captured: dict[str, Any] = {}
        self._inject: dict[str, torch.Tensor | None] = {"features": None}
        self._install_video_contract()

    def _install_video_contract(self) -> None:
        """Patch one private preprocessor instance, never global Molmo2 state."""
        preprocessor = self.preprocessor.video_preprocessor
        image_preprocessor = preprocessor.image_preprocessor
        original_load_video = preprocessor.load_video
        original_video_to_tokens = preprocessor.video_to_patches_and_tokens
        captured = self._captured

        def patched_load_video(video, clip=None, **kwargs):
            captured["video"] = str(video)
            captured["clip"] = clip
            if not captured.get("codec_active", False):
                # Source-frame-aligned OV2 comparisons use the same full-span
                # 1 FPS policy for native Dense and Codec inputs. Other modes
                # preserve Molmo2's untouched native Dense loader.
                if self.timeline_sampling_mode == "full_span_1fps":
                    dense_kwargs = dict(kwargs)
                    dense_kwargs.update(
                        _codec_timeline_sampler_overrides(
                            self.timeline_sampling_mode,
                            max_frames=self.max_frames,
                            base_fps=self.codec_config.base_fps,
                        )
                    )
                    return original_load_video(video, clip, **dense_kwargs)
                return original_load_video(video, clip, **kwargs)

            for cache_dir in self.gop_cache_read_dirs:
                cache_path = self._eval_selected_gop_cache_path(
                    cache_dir,
                    captured["video"],
                    captured["clip"],
                    self.codec_config,
                    timeline_max_frames=self.timeline_max_frames,
                    visual_token_budget=self.visual_token_budget,
                    n_p_tokens=self.n_p_tokens,
                )
                artifact = self._load_eval_selected_gop_artifact(
                    cache_path, self.codec_config
                )
                if artifact is None:
                    continue
                roles, selected, timestamps, u5 = artifact
                fast_frames = self._VideoFrames(
                    np.zeros((len(roles), 1, 1, 3), dtype=np.uint8),
                    timestamps,
                    self.codec_config.base_fps,
                    subtitle=kwargs.get("subtitle"),
                )
                fast_frames._molmo2_codec_artifact = (
                    roles,
                    selected,
                    u5,
                    cache_path,
                )
                return fast_frames

            timeline_kwargs = dict(kwargs)
            timeline_kwargs.update(
                _codec_timeline_sampler_overrides(
                    self.timeline_sampling_mode,
                    max_frames=self.timeline_max_frames,
                    base_fps=self.codec_config.base_fps,
                )
            )
            return original_load_video(video, clip, **timeline_kwargs)

        def apply_placeholder_contract(data):
            counts = self._evaluation_preprocessor_frame_counts(
                "codec",
                captured["roles"],
                n_p_tokens=self.n_p_tokens,
            )
            rewritten, local_trace = self._rewrite_tokenized_video_for_frame_counts(
                data,
                counts,
                image_patch_id=self.image_patch_token_id,
            )
            captured["placeholder_counts"] = tuple(counts)
            captured["local_position_trace"] = local_trace
            return rewritten

        def patched_video_to_tokens(
            frames, frame_prefixes, is_training=False, rng=None
        ):
            if not captured.get("codec_active", False):
                return original_video_to_tokens(
                    frames, frame_prefixes, is_training, rng
                )

            fast_artifact = getattr(frames, "_molmo2_codec_artifact", None)
            if fast_artifact is not None:
                roles, selected, u5, cache_path = fast_artifact
                captured["gop_roles"] = tuple(roles)
                captured["gop_source_indices"] = tuple(
                    int(x) for x in selected
                )
                if self.p_variant == "drop_p":
                    keep = np.asarray(_same_i_positions(roles), dtype=np.int64)
                    roles = tuple(roles[index] for index in keep)
                    selected = np.asarray(selected)[keep]
                    u5 = np.asarray(u5)[keep]
                    frames = self._VideoFrames(
                        frames.frames[keep],
                        np.asarray(frames.timestamps)[keep],
                        frames.target_fps,
                        frames.sampling_augmentation,
                        subtitle=frames.subtitle,
                    )
                    frame_prefixes = [
                        frame_prefixes[index] for index in keep
                    ]
                captured.update(
                    roles=tuple(roles),
                    source_indices=tuple(int(x) for x in selected),
                    u5=np.asarray(u5),
                    gop_cache_path=cache_path,
                    resized=None,
                    input_timestamps=tuple(
                        float(x) for x in np.asarray(frames.timestamps)
                    ),
                    input_frame_prefixes=tuple(str(x) for x in frame_prefixes),
                )
                return apply_placeholder_contract(
                    original_video_to_tokens(frames, frame_prefixes, is_training, rng)
                )

            timeline_timestamps = np.asarray(frames.timestamps, dtype=np.float64)
            timeline_resized = self._resize_eval_rgb_frames(
                image_preprocessor, frames.frames
            )
            motion_artifact = None
            for cache_dir in self.motion_cache_read_dirs:
                motion_path = self._eval_motion_cache_path(
                    cache_dir,
                    timeline_resized,
                    timeline_timestamps,
                    self.codec_config,
                )
                motion_artifact = self._load_eval_motion_cache(
                    motion_path,
                    len(timeline_resized),
                    self.codec_config,
                )
                if motion_artifact is not None:
                    break
            if motion_artifact is None:
                roles, motion, costs = self._codec_motion_roles(
                    timeline_resized, self.codec_config
                )
                if self.motion_cache_dir:
                    motion_path = self._eval_motion_cache_path(
                        self.motion_cache_dir,
                        timeline_resized,
                        timeline_timestamps,
                        self.codec_config,
                    )
                    self._atomic_save(
                        motion_path,
                        mv=motion.astype(np.float16),
                        cost=np.asarray(costs, dtype=np.float64),
                        schema=np.array(self.codec_config.schema_version),
                    )
            else:
                motion, costs = motion_artifact
                roles = self._assign_roles(
                    costs,
                    gamma=self.codec_config.require_fixed_gamma(),
                    max_gop=self.codec_config.gop_max_p,
                    target_p_per_gop=self.codec_config.gop_target_p,
                )

            selection = self._select_complete_gops(
                roles,
                self.visual_token_budget,
                n_i_tokens=self._N_I,
                n_p_tokens=self.n_p_tokens,
            )
            selected = np.asarray(selection.frame_indices, dtype=np.int64)
            selected_roles = tuple(roles[index] for index in selected)
            selected_u5 = self._selected_u5_from_motion(
                timeline_resized,
                roles,
                motion,
                selected,
                self.codec_config,
            )
            captured["gop_roles"] = selected_roles
            captured["gop_source_indices"] = tuple(
                int(x) for x in selected
            )
            if self.gop_cache_dir:
                cache_path = self._eval_selected_gop_cache_path(
                    self.gop_cache_dir,
                    captured["video"],
                    captured["clip"],
                    self.codec_config,
                    timeline_max_frames=self.timeline_max_frames,
                    visual_token_budget=self.visual_token_budget,
                    n_p_tokens=self.n_p_tokens,
                )
                self._atomic_save(
                    cache_path,
                    u5=self._codec_encode_u5(selected_u5),
                    roles=np.asarray(selected_roles),
                    ts=timeline_timestamps[selected],
                    source_indices=selected,
                    schema=np.array(self.codec_config.schema_version),
                )
                captured["gop_cache_path"] = cache_path

            if self.p_variant == "drop_p":
                keep = np.asarray(
                    _same_i_positions(selected_roles), dtype=np.int64
                )
                selected = selected[keep]
                selected_roles = tuple(
                    selected_roles[index] for index in keep
                )
                selected_u5 = selected_u5[keep]

            captured.update(
                roles=selected_roles,
                source_indices=tuple(int(x) for x in selected),
                u5=selected_u5,
                resized=self._selected_eval_rgb_frames(
                    image_preprocessor,
                    frames.frames,
                    selected,
                    timeline_resized=timeline_resized,
                ),
                input_timestamps=tuple(
                    float(x) for x in timeline_timestamps[selected]
                ),
            )
            frame_prefixes = [frame_prefixes[index] for index in selected]
            captured["input_frame_prefixes"] = tuple(
                str(x) for x in frame_prefixes
            )
            selected_frames = self._VideoFrames(
                frames.frames[selected],
                timeline_timestamps[selected],
                frames.target_fps,
                frames.sampling_augmentation,
                subtitle=frames.subtitle,
            )
            return apply_placeholder_contract(
                original_video_to_tokens(
                    selected_frames,
                    frame_prefixes,
                    is_training,
                    rng,
                )
            )

        preprocessor.load_video = patched_load_video
        preprocessor.video_to_patches_and_tokens = patched_video_to_tokens

        real_vision_forward = self.codec_runtime.forward_fn

        def injected_vision_forward(
            images, image_masks, pooled_patches_idx, enable_cp=False
        ):
            if self._inject["features"] is not None:
                return self._inject["features"]
            return real_vision_forward(
                images,
                image_masks,
                pooled_patches_idx,
                enable_cp=enable_cp,
            )

        self.model.vision_backbone.forward = injected_vision_forward

    @staticmethod
    def _make_batch(processed: dict[str, Any]) -> dict[str, torch.Tensor]:
        source_keys = {
            "input_tokens": "input_ids",
            "images": "images",
            "image_masks": "image_masks",
            "token_pooling": "token_pooling",
            "low_res_token_pooling": "low_res_token_pooling",
        }
        batch = {}
        for source, destination in source_keys.items():
            value = processed.get(source)
            if value is None:
                continue
            tensor = torch.from_numpy(np.asarray(value)[None])
            if destination in {
                "input_ids",
                "token_pooling",
                "low_res_token_pooling",
            }:
                tensor = tensor.long()
            batch[destination] = tensor
        if "input_ids" not in batch:
            raise ValueError("preprocessor emitted no input_tokens")
        return batch

    def _codec_features(self) -> tuple[torch.Tensor, list[int], Any]:
        captured = self._captured
        mode_by_variant = {
            "real": "codec",
            "drop_p": "codec_drop_p",
            "zero_output": "codec_zero_output",
            "shuffle_content": "codec_shuffle_content",
            "foreign": "codec_foreign_content",
        }
        cache_path = self._eval_frozen_feature_cache_path(
            self.feature_cache_dir,
            video=captured["video"],
            clip=captured.get("clip"),
            mode=mode_by_variant[self.p_variant],
            roles=captured["roles"],
            source_indices=captured["source_indices"],
            ptok_checkpoint=self.ptok_checkpoint,
            base_checkpoint=self.base_checkpoint,
            stage2_checkpoint=self.stage2_checkpoint,
            max_frames=self.max_frames,
            timeline_max_frames=self.timeline_max_frames,
            visual_token_budget=self.visual_token_budget,
            n_p_tokens=self.n_p_tokens,
            cfg=self.codec_config,
        )
        cached = None
        if self.p_variant != "foreign":
            cached = self._load_eval_frozen_feature_cache(
                cache_path, device=self.device
            )
        if cached is not None:
            return cached
        features, counts, trace = self.codec_runtime.features(
            captured["roles"],
            captured["u5"],
            p_variant=("real" if self.p_variant == "foreign" else self.p_variant),
            return_trace=True,
        )
        if self.p_variant != "foreign":
            self._store_eval_frozen_feature_cache(cache_path, features, counts, trace)
        return features, counts, trace

    def generate(
        self,
        *,
        prompt: str,
        visuals: list[Any],
        max_new_tokens: int,
        foreign_visuals: list[Any] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        kind = _visual_kind(visuals)
        example: dict[str, Any] = {
            "question": prompt,
            "style": "plain",
        }
        if kind == "video":
            example["video"] = os.fspath(visuals[0])
        elif kind == "image":
            example["image"] = visuals if len(visuals) > 1 else visuals[0]

        self._captured.clear()
        self._captured["codec_active"] = (
            kind == "video" and self.video_backend == "codec"
        )
        processed = self.preprocessor(example)
        metadata = processed.get("metadata", {})
        trace_payload = None
        if self.native_p_mode:
            batch = {
                key: value.to(self.device)
                for key, value in self._make_batch(processed).items()
            }
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                generated = self.model.generate(
                    batch=batch,
                    max_steps=int(max_new_tokens),
                    is_distributed=False,
                )
            prediction = self.config.post_process(
                batch, generated, [metadata]
            )["predictions_text"][0]
            return prediction, {
                "native_p_tokenization": self.p_tokenization,
                "video_backend": self.video_backend,
                "video": os.fspath(visuals[0]) if kind == "video" else None,
            }
        if self._captured["codec_active"]:
            if self.p_variant == "foreign":
                if _visual_kind(foreign_visuals or []) != "video":
                    raise ValueError("foreign P evaluation requires a donor video")
                target_capture = dict(self._captured)
                target_capture["u5"] = np.array(target_capture["u5"], copy=True)
                donor_video = os.fspath(foreign_visuals[0])
                if os.path.realpath(donor_video) == os.path.realpath(
                    os.fspath(visuals[0])
                ):
                    raise ValueError(
                        "foreign donor resolves to the target video itself"
                    )
                self._captured.clear()
                self._captured["codec_active"] = True
                self.preprocessor(
                    {
                        "video": donor_video,
                        "question": "Describe the video briefly.",
                        "style": "plain",
                    }
                )
                donor_capture = dict(self._captured)
                foreign_u5 = self._replace_p_u5_with_foreign(
                    target_capture["roles"],
                    target_capture["u5"],
                    donor_capture["roles"],
                    donor_capture["u5"],
                )
                self._captured.clear()
                self._captured.update(target_capture)
                self._captured["u5"] = foreign_u5
                self._captured["foreign_donor_video"] = donor_video
            features, counts, trace = self._codec_features()
            expected = tuple(self._captured["placeholder_counts"])
            actual_placeholders = int(
                np.count_nonzero(processed["input_tokens"] == self.image_patch_token_id)
            )
            if tuple(counts) != expected:
                raise AssertionError(
                    f"codec feature counts {tuple(counts)} != prompt {expected}"
                )
            if sum(counts) != actual_placeholders:
                raise AssertionError(
                    f"codec features {sum(counts)} != placeholders "
                    f"{actual_placeholders}"
                )
            self._inject["features"] = features.to(torch.bfloat16)
            trace_payload = {
                "codec": asdict(trace),
                "video": self._captured.get("video"),
                "source_indices": list(self._captured.get("source_indices", ())),
                "gop_roles": list(self._captured.get("gop_roles", ())),
                "gop_source_indices": list(
                    self._captured.get("gop_source_indices", ())
                ),
                "input_timestamps": list(
                    self._captured.get("input_timestamps", ())
                ),
                "input_frame_prefixes": list(
                    self._captured.get("input_frame_prefixes", ())
                ),
                "gop_cache_path": self._captured.get("gop_cache_path"),
                "p_variant": self.p_variant,
                "foreign_donor_video": self._captured.get("foreign_donor_video"),
            }

        batch = {
            key: value.to(self.device)
            for key, value in self._make_batch(processed).items()
        }
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = self.model.generate(
                    batch=batch,
                    max_steps=int(max_new_tokens),
                    is_distributed=False,
                )
            prediction = self.config.post_process(batch, generated, [metadata])[
                "predictions_text"
            ][0]
        finally:
            self._inject["features"] = None
        return prediction, trace_payload


@register_model("molmo2_codec")
class Molmo2Codec(lmms):
    """Native Molmo2/AdaCodec model exposed through lmms-eval."""

    def __init__(
        self,
        pretrained: str = "",
        training_repo: str = "/fsx/home/weikai.huang/molmo2_codec/mm_olmo",
        base_checkpoint: str | None = None,
        ptok_checkpoint: str = "",
        stage2_checkpoint: str | None = None,
        video_backend: str = "codec",
        visual_token_budget: int = 8192,
        timeline_max_frames: int = 2048,
        timeline_sampling_mode: str = "full_span_2fps",
        max_frames: int = 101,
        seq_len: int = 16384,
        gop_cache_dir: str = "",
        gop_cache_read_dirs: str = "",
        motion_cache_dir: str = "",
        motion_cache_read_dirs: str = "",
        feature_cache_dir: str = "",
        p_variant: str = "real",
        trace_output: str = "",
        gamma_artifact: str | None = None,
        device: str = "cuda",
        batch_size: int | str = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected model arguments: {sorted(kwargs)}")
        if int(batch_size) != 1:
            raise ValueError("molmo2_codec currently requires batch_size=1")
        # Accept lmms-eval's conventional `pretrained=` spelling as the Stage-2
        # checkpoint while retaining an explicit, less ambiguous argument.
        if (
            pretrained
            and stage2_checkpoint
            and os.path.abspath(pretrained) != os.path.abspath(stage2_checkpoint)
        ):
            raise ValueError(
                "pretrained and stage2_checkpoint refer to different files"
            )
        stage2_checkpoint = stage2_checkpoint or pretrained or None

        accelerator = Accelerator()
        self.accelerator = accelerator
        self._device = torch.device(
            f"cuda:{accelerator.local_process_index}"
            if accelerator.num_processes > 1
            else device
        )
        if self._device.type == "cuda":
            torch.cuda.set_device(self._device)
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes
        self.batch_size_per_gpu = 1
        self.pretrained = stage2_checkpoint or base_checkpoint or "molmo2"
        self._runtime = _Molmo2CodecRuntime(
            training_repo=training_repo,
            base_checkpoint=base_checkpoint,
            ptok_checkpoint=ptok_checkpoint,
            stage2_checkpoint=stage2_checkpoint,
            gamma_artifact=gamma_artifact,
            device=self._device,
            video_backend=video_backend,
            visual_token_budget=visual_token_budget,
            timeline_max_frames=timeline_max_frames,
            timeline_sampling_mode=timeline_sampling_mode,
            max_frames=max_frames,
            seq_len=seq_len,
            gop_cache_dir=gop_cache_dir,
            gop_cache_read_dirs=gop_cache_read_dirs,
            motion_cache_dir=motion_cache_dir,
            motion_cache_read_dirs=motion_cache_read_dirs,
            feature_cache_dir=feature_cache_dir,
            p_variant=p_variant,
        )
        self.last_codec_trace: dict[str, Any] | None = None
        self.trace_output = (
            _ranked_trace_path(
                os.path.abspath(os.path.expanduser(trace_output)),
                self._rank,
                self._world_size,
            )
            if trace_output
            else ""
        )

    @property
    def config(self):
        return self._runtime.config

    @property
    def tokenizer(self):
        return self._runtime.tokenizer

    @property
    def model(self):
        return self._runtime.model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._runtime.seq_len

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "molmo2_codec currently supports generative lmms-eval tasks"
        )

    def generate_until(self, requests: list[Instance]) -> list[str]:
        responses: list[str] = []
        prepared = []
        for request in requests:
            (
                context,
                generation_kwargs,
                doc_to_visual,
                doc_id,
                task,
                split,
            ) = request.args
            document = self.task_dict[task][split][doc_id]
            prepared.append(
                (
                    request,
                    context,
                    generation_kwargs,
                    doc_id,
                    task,
                    split,
                    _normalize_visuals(doc_to_visual(document)),
                )
            )
        if self._runtime.p_variant == "foreign" and len(prepared) < 2:
            raise ValueError(
                "foreign P evaluation requires at least two local requests"
            )
        progress = tqdm(
            enumerate(prepared),
            total=len(prepared),
            disable=self.rank != 0,
            desc="Molmo2/AdaCodec responding",
        )
        for index, item in progress:
            (
                request,
                context,
                generation_kwargs,
                doc_id,
                task,
                split,
                visuals,
            ) = item
            generation_kwargs = dict(generation_kwargs or {})
            if (
                generation_kwargs.get("do_sample", False)
                or float(generation_kwargs.get("temperature", 0.0) or 0.0) > 0
            ):
                raise ValueError(
                    "native Molmo2 adapter currently supports deterministic "
                    "generation only"
                )
            max_new_tokens = int(generation_kwargs.get("max_new_tokens", 128))
            response, trace = self._runtime.generate(
                prompt=_strip_lmms_placeholders(context),
                visuals=visuals,
                max_new_tokens=max_new_tokens,
                foreign_visuals=(
                    _select_foreign_visuals([entry[-1] for entry in prepared], index)
                    if self._runtime.p_variant == "foreign"
                    else None
                ),
            )
            self.last_codec_trace = trace
            responses.append(response)
            self.cache_hook.add_partial(
                "generate_until", (context, generation_kwargs), response
            )
            if self.trace_output:
                os.makedirs(os.path.dirname(self.trace_output), exist_ok=True)
                with open(self.trace_output, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "task": task,
                                "split": split,
                                "doc_id": doc_id,
                                "response": response,
                                "trace": trace,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        return responses

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError(
            "molmo2_codec does not yet support multi-round lmms-eval tasks"
        )
