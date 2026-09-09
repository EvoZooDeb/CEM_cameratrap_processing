from argparse import ArgumentParser
import os
from pathlib import Path
import signal

from cliff.command import Command

from ..config import load_config
from ..operations.aggregate import aggregate
from ..operations.classify import classify
from ..operations.exif import get_exif
from ..operations.predict import predict
from ..operations.validate import validate


class RunPipeline(Command):
    """Run prediction, metadata extraction, aggregation, and optional validation."""

    def get_parser(self, prog_name):
        parser = ArgumentParser(
            prog=prog_name,
            description=(
                "Run predict, exif, and aggregate in that order, then optionally validate "
                "the saved detections. On SIGTERM or when FELIS_CANCEL_FILE exists, the command "
                "writes EXIF and per-media summaries only for media completed before cancellation."
            ),
        )
        parser.add_argument("--config", help="YAML configuration file. CLI values override it.")
        parser.add_argument("--input-root", help="Base directory containing the raw survey data.")
        parser.add_argument(
            "--output-root", help="Base directory in which pipeline results are written."
        )
        parser.add_argument(
            "--username", help="Survey or project name used in the input and output paths."
        )
        parser.add_argument("--camera-id", help="Camera unit identifier to process.")
        parser.add_argument(
            "--footage-date", help="Camera footage date used to select the input directory."
        )
        parser.add_argument("--model-path", help="Path to the YOLO model weights file.")
        parser.add_argument("--strategy", choices=["single_stage", "two_stage"])
        parser.add_argument("--detector", choices=["best_27", "mdv6", "deepfaune_1.4", "best_28"])
        parser.add_argument(
            "--classifier",
            choices=["deepfaune_classifier", "4_camtrap", "2_artiodactyla", "2_carnivora"],
        )
        parser.add_argument(
            "--models-dir", help="Directory containing fixed two-stage model files."
        )
        parser.add_argument(
            "--device", default=None, help="Inference device, for example 'cuda:0' or 'cpu'."
        )
        parser.add_argument(
            "--imgsz", type=int, default=None, help="YOLO inference image size in pixels."
        )
        parser.add_argument(
            "--conf", type=float, default=None, help="Minimum YOLO detection confidence threshold."
        )
        parser.add_argument(
            "--iou",
            type=float,
            default=None,
            help="IoU threshold used by YOLO non-maximum suppression.",
        )
        parser.add_argument(
            "--save-frames",
            action="store_true",
            help=(
                "Save video frames during prediction so video detections can be visually "
                "validated."
            ),
        )
        parser.add_argument(
            "--save-per-image",
            action="store_true",
            help="Also write the per-media summary CSV during aggregation.",
        )
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Run visual validation after aggregation completes.",
        )
        parser.add_argument(
            "--no-show",
            action="store_true",
            help="Do not open validation windows; useful in headless environments.",
        )
        parser.add_argument(
            "--save-annotated",
            action="store_true",
            help="Write annotated validation overlays when --validate is also supplied.",
        )
        return parser

    def take_action(self, parsed_args):
        overrides = {
            "input_root": parsed_args.input_root,
            "output_root": parsed_args.output_root,
            "username": parsed_args.username,
            "camera_id": parsed_args.camera_id,
            "footage_date": parsed_args.footage_date,
            "model_path": parsed_args.model_path,
            "device": parsed_args.device,
            "imgsz": parsed_args.imgsz,
            "conf": parsed_args.conf,
            "iou": parsed_args.iou,
            "save_frames": parsed_args.save_frames,
            "strategy": parsed_args.strategy,
            "detector": parsed_args.detector,
            "classifier": parsed_args.classifier,
            "models_dir": parsed_args.models_dir,
        }
        cfg = load_config(parsed_args.config, overrides)

        cancellation_requested = False
        cancel_file = os.environ.get("FELIS_CANCEL_FILE")

        def cancellation_is_requested():
            return cancellation_requested or (cancel_file and Path(cancel_file).exists())

        def request_cancellation(_signum, _frame):
            nonlocal cancellation_requested
            cancellation_requested = True
            self.app.stdout.write("Cancellation requested; finalizing completed media only...\n")

        previous_sigterm_handler = signal.signal(signal.SIGTERM, request_cancellation)
        try:
            steps = 4 if cfg.two_stage.strategy == "two_stage" else 3
            self.app.stdout.write(f"[1/{steps}] Predicting...\n")
            completed_files = predict(cfg, should_cancel=cancellation_is_requested)

            if cancellation_is_requested():
                self.app.stdout.write("[cancelled] Writing partial EXIF and results...\n")
                get_exif(cfg, include_files=set(completed_files))
                classify(cfg, completed_files=completed_files)
                aggregate(cfg, save_per_image=True, completed_files=completed_files)
                return

            self.app.stdout.write(f"[2/{steps}] Extracting EXIF...\n")
            get_exif(cfg)

            if cfg.two_stage.strategy == "two_stage":
                self.app.stdout.write("[3/4] Classifying detected animals...\n")
                classify(cfg)

            self.app.stdout.write(f"[{steps}/{steps}] Aggregating...\n")
            aggregate(cfg, save_per_image=parsed_args.save_per_image)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if parsed_args.validate:
            self.app.stdout.write("[+] Validating (visual) ...\n")
            validate(cfg, show=not parsed_args.no_show, save_annotated=parsed_args.save_annotated)
