from argparse import ArgumentParser

from cliff.command import Command

from ..config import load_config
from ..core import get_exif


class GetExif(Command):
    """Extract capture timestamps and durations into the EXIF CSV."""

    def get_parser(self, prog_name):
        parser = ArgumentParser(
            prog=prog_name,
            description=(
                "Create the camera-date EXIF CSV from the input directory. JPG and PNG "
                "timestamps come from EXIF when available, otherwise file modification time; "
                "MP4 and MOV timestamps and durations come from container metadata, with the "
                "same modification-time fallback. AVI files are counted and skipped."
            ),
        )
        parser.add_argument("--config", help="YAML configuration file. CLI values override it.")
        parser.add_argument("--input-root", help="Base directory containing the raw survey data.")
        parser.add_argument(
            "--output-root", help="Base directory in which the EXIF CSV is written."
        )
        parser.add_argument(
            "--username", help="Survey or project name used in the input and output paths."
        )
        parser.add_argument("--camera-id", help="Camera unit identifier to process.")
        parser.add_argument(
            "--footage-date", help="Camera footage date used to select the input directory."
        )
        parser.add_argument(
            "--model-path",
            help=(
                "Path to YOLO weights; required by the shared configuration although EXIF "
                "extraction does not use it."
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
        get_exif(cfg)
