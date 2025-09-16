from argparse import ArgumentParser

from cliff.command import Command

from ..config import load_config
from ..core import get_exif


class GetExif(Command):
    """Extract EXIF/metadata to CSV."""

    def get_parser(self, prog_name):
        parser = ArgumentParser(prog=prog_name)
        parser.add_argument("--config", help="Path to YAML config file")
        parser.add_argument("--input-root")
        parser.add_argument("--output-root")
        parser.add_argument("--username")
        parser.add_argument("--camera-id")
        parser.add_argument("--footage-date")
        parser.add_argument("--model-path")  # required by config, but not used here
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
        df, avi_count = get_exif(cfg)
        self.app.stdout.write(f"Extracted {len(df)} records. AVI skipped: {avi_count}\n")

