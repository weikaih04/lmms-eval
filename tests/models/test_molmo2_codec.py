import inspect
import dataclasses
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TRAINING_REPO = Path("/fsx/home/weikai.huang/molmo2_codec/mm_olmo")
if str(TRAINING_REPO) not in sys.path:
    sys.path.insert(0, str(TRAINING_REPO))

from olmo.data.video_loader import FrameSampler, TimeSampler

from lmms_eval.models.simple.molmo2_codec import (
    Molmo2Codec,
    _codec_timeline_sampler_overrides,
    _normalize_visuals,
    _ranked_trace_path,
    _select_foreign_visuals,
    _strip_lmms_placeholders,
    _visual_kind,
)


def test_full_span_timeline_uses_molmo_uniform_fallback_contract():
    assert _codec_timeline_sampler_overrides(
        "full_span_2fps", max_frames=2048, base_fps=2.0
    ) == {
        "max_frames": 2048,
        "frame_sample_mode": "uniform_last_frame",
        "max_fps": 2.0,
        "min_fps": 2.0,
    }


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
