from __future__ import annotations

import io
import logging
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from felis_cli.config import Config, Paths, PredictParams, TwoStageParams
from felis_cli.operations import classify as classify_operation
from felis_cli.operations import predict as predict_operation
from felis_cli.operations.paths import resolve_paths
from felis_cli.verbosity import backend_verbose, configure, diagnostic, third_party_stdout


class VerbosityTests(unittest.TestCase):
    def tearDown(self) -> None:
        configure(1)

    def test_third_party_stdout_is_hidden_until_double_verbose(self) -> None:
        for level, expected in ((0, ""), (1, ""), (2, ""), (3, "backend\n")):
            with self.subTest(level=level):
                configure(level)
                output = io.StringIO()
                with redirect_stdout(output), third_party_stdout():
                    print("backend")
                self.assertEqual(expected, output.getvalue())

    def test_diagnostics_require_double_verbose(self) -> None:
        logger = logging.getLogger("felis.tests.verbosity")
        with patch.object(logger, "debug") as debug:
            configure(2)
            diagnostic(logger, "hidden")
            debug.assert_not_called()
            configure(3)
            diagnostic(logger, "visible")
            debug.assert_called_once_with("visible")

    def test_ultralytics_info_messages_follow_verbosity(self) -> None:
        logger = logging.getLogger("ultralytics")
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        with patch.object(logger, "handlers", [handler]):
            configure(0)
            backend_verbose()
            logger.info("Results saved to hidden/path")
            logger.warning("visible warning")
            self.assertEqual("visible warning\n", output.getvalue())

            output.seek(0)
            output.truncate(0)
            configure(2)
            backend_verbose()
            logger.info("Results saved to visible/path")
            self.assertEqual("Results saved to visible/path\n", output.getvalue())


class CoreProgressTests(unittest.TestCase):
    def _config(self, root: Path, strategy: str = "single_stage") -> Config:
        model_path = root / "model.pt"
        model_path.touch()
        input_root = root / "raw"
        input_dir = input_root / "survey" / "files" / "cameratrap" / "camera" / "20260101"
        input_dir.mkdir(parents=True)
        return Config(
            paths=Paths(input_root, root / "results", "survey", "camera", "20260101"),
            predict=PredictParams(model_path=model_path, device="cpu"),
            two_stage=TwoStageParams(strategy=strategy, models_dir=root),
        )

    def test_predict_controls_ultralytics_verbosity(self) -> None:
        class FakeModel:
            calls: list[dict[str, object]] = []

            def __init__(self, _path: str) -> None:
                pass

            def predict(self, _source: str, **kwargs: object) -> list[object]:
                self.calls.append(kwargs)
                return []

        with tempfile.TemporaryDirectory() as temporary_directory:
            cfg = self._config(Path(temporary_directory))
            (resolve_paths(cfg).input_dir / "sample.jpg").touch()
            (resolve_paths(cfg).input_dir / "sample.mp4").touch()
            with patch.object(predict_operation, "YOLO", FakeModel):
                configure(1)
                predict_operation.predict(cfg)
                configure(2)
                predict_operation.predict(cfg)

        self.assertTrue(all(not call["verbose"] for call in FakeModel.calls[:2]))
        self.assertTrue(all(call["verbose"] for call in FakeModel.calls[2:]))
        self.assertTrue(all("augment" not in call for call in FakeModel.calls))

    def test_predict_saves_video_frames_for_two_stage_classification(self) -> None:
        class FakeResult:
            orig_img = object()

        class FakeModel:
            def __init__(self, _path: str) -> None:
                pass

            def predict(self, source: str, **_kwargs: object) -> list[FakeResult]:
                return [FakeResult()] if source.endswith(".mp4") else []

        with tempfile.TemporaryDirectory() as temporary_directory:
            cfg = self._config(Path(temporary_directory), strategy="two_stage")
            cfg.two_stage.detector = "best_27"
            (cfg.two_stage.models_dir / "best_27.pt").touch()
            (resolve_paths(cfg).input_dir / "sample.mp4").touch()
            with (
                patch.object(predict_operation, "YOLO", FakeModel),
                patch.object(
                    predict_operation.cv2,
                    "imwrite",
                    return_value=True,
                    create=True,
                ) as imwrite,
            ):
                predict_operation.predict(cfg)

        frame_path = resolve_paths(cfg).per_file_root / "sample" / "sample_frames/sample_1.jpg"
        imwrite.assert_called_once_with(str(frame_path), FakeResult.orig_img)

    def test_classify_reports_media_progress_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cfg = self._config(Path(temporary_directory), strategy="two_stage")
            (resolve_paths(cfg).input_dir / "sample.jpg").touch()
            with (
                patch.object(
                    classify_operation,
                    "_load_classifier",
                    return_value=(lambda image: ("x", 1.0), "fake"),
                ),
                self.assertLogs("felis_cli.operations.classify", level="INFO") as messages,
            ):
                classify_operation.classify(cfg)

        output = "\n".join(messages.output)
        self.assertIn("Classifying [1/1]: sample.jpg", output)
        self.assertIn("Classification complete", output)


if __name__ == "__main__":
    unittest.main()
