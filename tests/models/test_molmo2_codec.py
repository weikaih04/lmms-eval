import inspect
import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

TRAINING_REPO_ENV = os.environ.get("MOLMO2_CODEC_REPO")
TRAINING_REPO = Path(TRAINING_REPO_ENV) if TRAINING_REPO_ENV else None
if TRAINING_REPO is not None and TRAINING_REPO.is_dir() and str(TRAINING_REPO) not in sys.path:
    sys.path.insert(0, str(TRAINING_REPO))

try:
    from olmo.data.video_loader import FrameSampler, TimeSampler
except ImportError:
    FrameSampler = TimeSampler = None

from lmms_eval.models.simple.molmo2_codec import (
    Molmo2Codec,
    _Molmo2CodecRuntime,
    _codec_timeline_sampler_overrides,
    _same_i_positions,
    _normalize_visuals,
    _ranked_trace_path,
    _select_foreign_visuals,
    _strip_lmms_placeholders,
    _visual_kind,
)


def test_native_p_dispatch_precedes_historical_ptokenizer_runtime():
    source = inspect.getsource(_Molmo2CodecRuntime.__init__)
    assert source.index("if self.native_p_mode") < source.index(
        "CodecInferenceRuntime.from_checkpoints"
    )
    assert "_load_native_model" in source
    assert "video_preprocessor.video_backend" in source
    assert source.index("if self.native_p_mode") < source.index(
        "ptok_checkpoint is required for the historical learned"
    )


def test_public_adapter_does_not_require_dummy_ptokenizer_for_native_mode():
    source = inspect.getsource(Molmo2Codec.__init__)
    assert 'raise ValueError("ptok_checkpoint is required")' not in source


def test_model_batch_preserves_native_visual_contract_fields():
    processed = {
        "input_tokens": np.asarray([1, 2], dtype=np.int32),
        "images": np.zeros((2, 3), dtype=np.float32),
        "image_masks": np.ones((2,), dtype=np.bool_),
        "token_pooling": np.asarray([0, 1], dtype=np.int32),
        "low_res_token_pooling": np.asarray([1, 0], dtype=np.int32),
    }
    batch = _Molmo2CodecRuntime._make_batch(processed)
    assert set(batch) == {
        "input_ids",
        "images",
        "image_masks",
        "token_pooling",
        "low_res_token_pooling",
    }
    assert batch["input_ids"].dtype == torch.int64
    assert batch["token_pooling"].dtype == torch.int64
    assert batch["low_res_token_pooling"].dtype == torch.int64


def test_drop_p_keeps_only_exact_i_positions():
    assert _same_i_positions(("I", "P", "P", "I", "P")) == (0, 3)


def test_drop_p_rejects_invalid_role_contract():
    with pytest.raises(ValueError, match="start with an I-frame"):
        _same_i_positions(("P", "I"))


def test_full_span_timeline_uses_molmo_uniform_fallback_contract():
    assert _codec_timeline_sampler_overrides(
        "full_span_2fps", max_frames=2048, base_fps=2.0
    ) == {
        "max_frames": 2048,
        "frame_sample_mode": "uniform_last_frame",
        "max_fps": 2.0,
        "min_fps": 2.0,
    }


def test_ov2_aligned_timeline_uses_full_span_one_fps_contract():
    assert _codec_timeline_sampler_overrides(
        "full_span_1fps", max_frames=512, base_fps=2.0
    ) == {
        "max_frames": 512,
        "frame_sample_mode": "uniform_last_frame",
        "max_fps": 1.0,
        "min_fps": 1.0,
    }


def test_ov2_aligned_timeline_caps_long_video_but_keeps_full_span():
    overrides = _codec_timeline_sampler_overrides(
        "full_span_1fps", max_frames=512, base_fps=2.0
    )
    sampler = _apply_supported_overrides(TimeSampler(), overrides)
    _, times, _ = sampler(1800.0)
    assert len(times) == 512
    assert times[0] == 0.0 and times[-1] == 1800.0


def test_prefix_timeline_remains_an_explicit_legacy_mode():
    assert _codec_timeline_sampler_overrides(
        "prefix_2fps", max_frames=2048, base_fps=2.0
    ) == {
        "max_frames": 2048,
        "frame_sample_mode": "fps",
        "candidate_sampling_fps": (2.0,),
    }


def test_unknown_timeline_mode_is_rejected():
    try:
        _codec_timeline_sampler_overrides(
            "first_frames", max_frames=2048, base_fps=2.0
        )
    except ValueError as error:
        assert "timeline_sampling_mode" in str(error)
    else:
        raise AssertionError("unknown timeline modes must fail closed")


def _apply_supported_overrides(sampler, overrides):
    fields = {field.name for field in dataclasses.fields(sampler)}
    return dataclasses.replace(
        sampler, **{key: value for key, value in overrides.items() if key in fields}
    )


@pytest.mark.skipif(TimeSampler is None, reason="Molmo2 training repository is unavailable")
def test_full_span_contract_covers_short_and_long_time_sampled_videos():
    overrides = _codec_timeline_sampler_overrides(
        "full_span_2fps", max_frames=2048, base_fps=2.0
    )
    sampler = _apply_supported_overrides(TimeSampler(), overrides)
    _, short_times, _ = sampler(600.0)
    _, long_times, _ = sampler(1800.0)
    assert len(short_times) == 1201
    assert short_times[0] == 0.0 and short_times[-1] == 600.0
    assert len(long_times) == 2048
    assert long_times[0] == 0.0 and long_times[-1] == 1800.0


@pytest.mark.skipif(FrameSampler is None, reason="Molmo2 training repository is unavailable")
def test_full_span_contract_covers_long_frame_sampled_videos():
    overrides = _codec_timeline_sampler_overrides(
        "full_span_2fps", max_frames=2048, base_fps=2.0
    )
    sampler = _apply_supported_overrides(FrameSampler(), overrides)
    _, frame_indices, _ = sampler(video_fps=30.0, total_frames=54_000)
    assert len(frame_indices) == 2048
    assert frame_indices[0] == 0
    assert frame_indices[-1] == 53_999


def test_adapter_satisfies_lmms_abstract_interface():
    assert not inspect.isabstract(Molmo2Codec)


def test_trace_path_is_rank_local_for_data_parallel_eval():
    assert _ranked_trace_path("trace.jsonl", 0, 1) == "trace.jsonl"
    assert _ranked_trace_path("trace.jsonl", 3, 8) == ("trace.jsonl.rank3.jsonl")


def test_foreign_selector_skips_qa_rows_from_same_video(tmp_path):
    target = tmp_path / "target.mp4"
    donor = tmp_path / "donor.mp4"
    visuals = [[target], [target], [donor]]
    assert _select_foreign_visuals(visuals, 0) == [donor]


def test_foreign_selector_rejects_no_distinct_donor(tmp_path):
    target = tmp_path / "target.mp4"
    try:
        _select_foreign_visuals([[target], [target]], 0)
    except ValueError as error:
        assert "distinct donor" in str(error)
    else:
        raise AssertionError("same-video donors must be rejected")


def test_strip_lmms_placeholders_preserves_question():
    prompt = "<video> What happens after <image 2> the door opens?"
    assert _strip_lmms_placeholders(prompt) == ("What happens after  the door opens?")


def test_visual_kind_accepts_one_video_path():
    assert _visual_kind([Path("clip.mp4")]) == "video"


def test_visual_kind_accepts_native_images():
    pil_image = Image.new("RGB", (2, 2))
    array_image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert _visual_kind([pil_image, array_image]) == "image"


def test_normalize_visuals_flattens_one_level():
    assert _normalize_visuals([["a.png"], ["b.png"]]) == [
        "a.png",
        "b.png",
    ]


def test_multiple_videos_are_rejected():
    try:
        _visual_kind(["a.mp4", "b.mp4"])
    except ValueError as error:
        assert "one video" in str(error)
    else:
        raise AssertionError("multiple videos should be rejected")
