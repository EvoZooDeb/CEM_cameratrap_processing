import json
import zipfile
from pathlib import Path

from felis_cli.config import Config, Paths, PredictParams, TwoStageParams
from felis_cli.operations.artifacts import package_artifacts


def test_package_artifacts_excludes_frames_verifies_and_cleans(tmp_path: Path):
    config = Config(
        paths=Paths(tmp_path / "input", tmp_path / "output", "alice", "CAM-1", "survey"),
        predict=PredictParams(tmp_path / "model.pt"),
        two_stage=TwoStageParams(),
    )
    analysis_dir = (
        config.paths.output_root
        / config.paths.username
        / config.paths.camera_id
        / config.paths.footage_date
    )
    detections = analysis_dir / "results" / "detections"
    detections.mkdir(parents=True)
    (analysis_dir / "results" / "summary.csv").write_text("label\nfox\n")
    (detections / "one.json").write_text('{"label":"fox"}')
    frames = analysis_dir / "clip" / "clip_frames"
    frames.mkdir(parents=True)
    (frames / "frame.jpg").write_bytes(b"frame")
    (analysis_dir / "annotated.jpg").write_bytes(b"image")

    archive_path = package_artifacts(config)

    assert list(analysis_dir.iterdir()) == [archive_path]
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        assert "results/summary.csv" in names
        assert "results/detections/one.json" in names
        assert "results/detections.zip" in names
        assert "annotated.jpg" in names
        assert all("_frames/" not in name for name in names)
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["video_frames_included"] is False
        assert {entry["path"] for entry in manifest["files"]} == names - {"manifest.json"}
