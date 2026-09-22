import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from felis_cli.config import Config, Paths, PredictParams, TwoStageParams
from felis_cli.operations.aggregate import aggregate
from felis_cli.operations.paths import resolve_paths


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.cfg = Config(
            paths=Paths(root / "raw", root / "output", "user", "camera", "analysis"),
            predict=PredictParams(root / "model.pt"),
            two_stage=TwoStageParams(strategy="single_stage"),
        )
        self.paths = resolve_paths(self.cfg)
        self.paths.input_dir.mkdir(parents=True)
        self.paths.results_dir.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def _write_media(self, name, timestamp, duration=1.0, labels=None):
        (self.paths.input_dir / name).write_bytes(b"media")
        stem = Path(name).stem
        if labels is not None:
            label_dir = self.paths.per_file_root / stem / "labels"
            label_dir.mkdir(parents=True, exist_ok=True)
            lines = [
                f"{class_id} 0.5 0.5 0.2 0.2 {confidence}"
                for class_id, confidence in labels
            ]
            (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        return {"file_name": name, "date_exif": timestamp, "duration": duration}

    def test_emits_one_media_and_event_row_per_label(self):
        metadata = [
            self._write_media(
                "fox-and-deer.JPG",
                "2026:09:16 08:32:40",
                labels=[(2, 0.9), (5, 0.8), (5, 0.7)],
            )
        ]
        pd.DataFrame(metadata).to_csv(self.paths.media_metadata_csv, index=False)

        result = aggregate(self.cfg)

        self.assertEqual(["Cervus elaphus", "Vulpes vulpes"], result.media["label"].tolist())
        fox = result.media[result.media["label"] == "Vulpes vulpes"].iloc[0]
        self.assertEqual(2, fox["detection_count"])
        self.assertEqual(1, fox["annotated_frame_count"])
        self.assertEqual(2, fox["max_group_size"])
        self.assertAlmostEqual(0.75, fox["mean_detector_confidence"])
        self.assertTrue(pd.isna(fox["mean_classification_confidence"]))
        self.assertEqual({"Cervus elaphus", "Vulpes vulpes"}, set(result.events["label"]))

        detection_file = next(self.paths.detections_dir.glob("*.json"))
        document = json.loads(detection_file.read_text(encoding="utf-8"))
        self.assertEqual(2, document["schema_version"])
        self.assertIn("detector_confidence", document["detections"][0])
        self.assertIn("classification_confidence", document["detections"][0])
        self.assertNotIn("confidence", document["detections"][0])

    def test_includes_media_without_a_label_directory_as_empty(self):
        metadata = [self._write_media("empty.JPG", "2026:09:16 08:32:40")]
        pd.DataFrame(metadata).to_csv(self.paths.media_metadata_csv, index=False)

        result = aggregate(self.cfg)

        self.assertEqual("EMPTY", result.media.iloc[0]["label"])
        self.assertEqual(0, result.media.iloc[0]["detection_count"])
        self.assertEqual("EMPTY", result.events.iloc[0]["label"])

    def test_event_statistics_are_species_specific_across_media(self):
        metadata = [
            self._write_media("first.JPG", "2026:09:16 08:32:40", labels=[(5, 0.8)]),
            self._write_media(
                "second.JPG", "2026:09:16 08:32:45", labels=[(5, 0.6), (2, 0.9)]
            ),
        ]
        pd.DataFrame(metadata).to_csv(self.paths.media_metadata_csv, index=False)

        result = aggregate(self.cfg)

        fox = result.events[result.events["label"] == "Vulpes vulpes"].iloc[0]
        deer = result.events[result.events["label"] == "Cervus elaphus"].iloc[0]
        self.assertEqual(2, fox["media_count"])
        self.assertEqual(2, fox["detection_count"])
        self.assertEqual(1, deer["media_count"])
        self.assertEqual(["first.JPG", "second.JPG"], json.loads(fox["media_files"]))

    def test_separates_classifier_confidence_from_detector_confidence(self):
        metadata = [
            self._write_media("classified.JPG", "2026:09:16 08:32:40", labels=[(0, 0.91)])
        ]
        pd.DataFrame(metadata).to_csv(self.paths.media_metadata_csv, index=False)
        self.cfg.two_stage.strategy = "two_stage"
        self.paths.two_stage_json.write_text(
            json.dumps({
                "classified/labels/classified.txt:0": {
                    "label": "Vulpes_vulpes",
                    "confidence": 0.73,
                    "source": "classifier",
                }
            }),
            encoding="utf-8",
        )

        result = aggregate(self.cfg)

        row = result.media.iloc[0]
        self.assertAlmostEqual(0.91, row["mean_detector_confidence"])
        self.assertAlmostEqual(0.73, row["mean_classification_confidence"])


if __name__ == "__main__":
    unittest.main()
