from lmms_eval.tasks.mlvu.utils import (
    _molmo2_mlvu_video_index,
    _resolve_molmo2_mlvu_video,
)


def test_local_mlvu_index_maps_unique_basenames(tmp_path):
    category = tmp_path / "category"
    category.mkdir()
    video = category / "sample.mp4"
    video.write_bytes(b"test")
    assert _molmo2_mlvu_video_index(str(tmp_path)) == {"sample.mp4": [str(video)]}


def test_local_mlvu_index_keeps_duplicate_basenames_by_category(tmp_path):
    for category_name in ("one", "two"):
        category = tmp_path / category_name
        category.mkdir()
        (category / "sample.mp4").write_bytes(b"test")
    index = _molmo2_mlvu_video_index(str(tmp_path))
    assert len(index["sample.mp4"]) == 2


def test_local_mlvu_resolver_uses_task_category(tmp_path, monkeypatch):
    for category_name in ("plotQA", "topic_reasoning"):
        category = tmp_path / category_name
        category.mkdir()
        (category / "sample.mp4").write_bytes(b"test")
    monkeypatch.setenv("MLVU_VIDEO_ROOT", str(tmp_path))
    assert _resolve_molmo2_mlvu_video(
        {"video_name": "sample.mp4", "task_type": "plotQA"}
    ) == str(tmp_path / "plotQA" / "sample.mp4")
