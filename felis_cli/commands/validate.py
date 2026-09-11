from argparse import ArgumentParser

from cliff.command import Command

from ..config import load_config
from ..operations.validate import validate


class Validate(Command):
    """Review saved detections as OpenCV overlays or annotated image files."""

    def get_parser(self, prog_name):
        parser = ArgumentParser(
            prog=prog_name,
            description=(
                "Draw saved YOLO detections above 0.25 confidence on still images or saved "
                "video frames. Images can be displayed one at a time in OpenCV windows and/or "
                "written as annotated files. Videos can be reviewed only when predict was run "
                "with --save-frames."
            ),
        )
        parser.add_argument("--config", help="YAML configuration file. CLI values override it.")
        parser.add_argument("--input-root", help="Base directory containing the raw survey data.")
        parser.add_argument("--output-root", help="Base directory containing prediction outputs.")
        parser.add_argument(
            "--username", help="Survey or project name used in the input and output paths."
        )
        parser.add_argument("--camera-id", help="Camera unit identifier to review.")
        parser.add_argument(
            "--footage-date", help="Camera footage date used to select the input directory."
        )
        parser.add_argument(
            "--model-path",
            help=(
                "Path to YOLO weights; required by the shared configuration although "
                "validation does not use it."
            ),
        )
        parser.add_argument(
            "--no-show",
            action="store_true",
            help="Do not open OpenCV windows; use with --save-annotated for headless output.",
        )
        parser.add_argument(
            "--save-annotated",
            action="store_true",
            help=(
                "Write overlays to <output-root>/<username>/<camera-id>/<footage-date>/"
                "<media>/annotated."
            ),
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
        }
        cfg = load_config(parsed_args.config, overrides)
        validate(cfg, show=not parsed_args.no_show, save_annotated=parsed_args.save_annotated)
