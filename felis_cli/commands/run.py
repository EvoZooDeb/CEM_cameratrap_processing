from argparse import ArgumentParser
import os
from pathlib import Path
import signal

from cliff.command import Command

from ..config import load_config
from ..core import aggregate, get_exif, predict, validate


class RunPipeline(Command):
    """Run the pipeline: predict -> exif -> aggregate (optional validate)."""

    def get_parser(self, prog_name):
        parser = ArgumentParser(prog=prog_name)
        parser.add_argument("--config", help="Path to YAML config file")
        parser.add_argument("--input-root")
        parser.add_argument("--output-root")
        parser.add_argument("--username")
        parser.add_argument("--camera-id")
        parser.add_argument("--footage-date")
        parser.add_argument("--model-path")
        parser.add_argument("--device", default=None)
        parser.add_argument("--imgsz", type=int, default=None)
        parser.add_argument("--conf", type=float, default=None)
        parser.add_argument("--iou", type=float, default=None)
        parser.add_argument("--save-frames", action="store_true")
        parser.add_argument(
            "--save-per-image",
            action="store_true",
            help="Also write per-image summary CSV during aggregate",
        )
        parser.add_argument("--validate", action="store_true", help="Run visual validation at the end")
        parser.add_argument("--no-show", action="store_true", help="Do not show windows during validate")
        parser.add_argument(
            "--save-annotated",
            action="store_true",
            help="Save annotated images during validate",
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
            self.app.stdout.write("[1/3] Predicting...\n")
            completed_files = predict(cfg, should_cancel=cancellation_is_requested)

            if cancellation_is_requested():
                self.app.stdout.write("[cancelled] Writing partial EXIF and results...\n")
                get_exif(cfg, include_files=set(completed_files))
                aggregate(cfg, save_per_image=True, completed_files=completed_files)
                return

            self.app.stdout.write("[2/3] Extracting EXIF...\n")
            get_exif(cfg)

            self.app.stdout.write("[3/3] Aggregating...\n")
            aggregate(cfg, save_per_image=parsed_args.save_per_image)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if parsed_args.validate:
            self.app.stdout.write("[+] Validating (visual) ...\n")
            validate(cfg, show=not parsed_args.no_show, save_annotated=parsed_args.save_annotated)
