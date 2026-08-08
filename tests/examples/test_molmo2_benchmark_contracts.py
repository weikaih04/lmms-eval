import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmms_eval.api.task import Task


ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples" / "molmo2_codec"
MANIFEST = (
    ROOT
    / "examples"
    / "llava_onevision2_repro"
    / "manifests"
    / "VIDEO_QUALIFICATION_RANDOM200_V2_MANIFEST.json"
)


def test_random200_v2_manifest_is_frozen_and_complete():
    data = json.loads(MANIFEST.read_text())
    expected_sizes = {
        "ov2_videomme_short_wo_sutitle": 900,
        "ov2_videomme_medium_wo_sutitle": 900,
        "ov2_videomme_long_wo_sutitle": 900,
        "mlvu_dev": 2174,
    }
    assert set(data["tasks"]) == set(expected_sizes)
    for task_name, full_size in expected_sizes.items():
        task = data["tasks"][task_name]
        doc_ids = task["doc_ids"]
        assert task["full_size"] == full_size
        assert task["sample_size"] == 200
        assert len(doc_ids) == len(set(doc_ids)) == 200
        assert doc_ids == sorted(doc_ids)
        assert min(doc_ids) >= 0 and max(doc_ids) < full_size
        payload = json.dumps(doc_ids, separators=(",", ":")).encode()
        assert hashlib.sha256(payload).hexdigest() == task["sorted_id_sha256"]


def test_task_manifest_loader_returns_task_native_ids(monkeypatch):
    monkeypatch.setenv("LMMS_EVAL_DOC_ID_MANIFEST", str(MANIFEST))
    dummy = SimpleNamespace(
        config=SimpleNamespace(task="ov2_videomme_long_wo_sutitle"),
        eval_docs=[{}] * 900,
    )
    selected = Task._manifest_doc_ids(dummy)
    expected = json.loads(MANIFEST.read_text())["tasks"][
        "ov2_videomme_long_wo_sutitle"
    ]["doc_ids"]
    assert selected == expected


def test_task_manifest_loader_rejects_dataset_drift(monkeypatch):
    monkeypatch.setenv("LMMS_EVAL_DOC_ID_MANIFEST", str(MANIFEST))
    dummy = SimpleNamespace(
        config=SimpleNamespace(task="ov2_videomme_long_wo_sutitle"),
        eval_docs=[{}] * 899,
    )
    with pytest.raises(ValueError, match="Dataset size drift"):
        Task._manifest_doc_ids(dummy)


def test_molmo_runner_has_controlled_and_ov2_policy_modes():
    text = (EXAMPLES / "run_paired_video.sh").read_text()
    assert "FRAME_CAP=384" in text
    assert "TIMELINE_MODE=full_span_2fps" in text
    assert "TIMELINE_MODE=full_span_1fps" in text
    assert "visual_token_budget=${VISUAL_TOKEN_BUDGET}" in text
    assert "VISUAL_TOKEN_BUDGET=${VISUAL_TOKEN_BUDGET:-8192}" in text
    assert "LMMS_EVAL_DOC_ID_MANIFEST" in text
    assert "BACKENDS=${BACKENDS:-dense,codec}" in text
    assert "/fsx/" not in text
    assert "--limit" not in text


def test_ov2_runner_records_official_and_molmo384_settings():
    text = (EXAMPLES / "run_ov2_reference.sh").read_text()
    for setting in (
        "OFFICIAL_F=128; MP=321489",
        "OFFICIAL_F=256; MP=136900",
        "OFFICIAL_F=640; MP=102400",
        "OFFICIAL_F=512; MP=72900",
        "FRAME_CAP=384; FPS=2",
    ):
        assert setting in text
    assert "allow_all_kernels=True" in text
    assert "LMMS_EVAL_DOC_ID_MANIFEST" in text
    assert "/fsx/" not in text
    assert "--limit" not in text
