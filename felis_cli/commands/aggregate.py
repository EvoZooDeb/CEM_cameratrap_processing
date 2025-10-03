from argparse import ArgumentParser

from cliff.command import Command

from ..config import load_config
from ..core import aggregate


class Aggregate(Command):
    """Create per-sequence summary CSV from labels + EXIF."""

    def get_parser(self, prog_name):
        parser = ArgumentParser(prog=prog_name)
        parser.add_argument("--config", help="Path to YAML config file")
        parser.add_argument("--input-root")
        parser.add_argument("--output-root")
        parser.add_argument("--username")
        parser.add_argument("--camera-id")
        parser.add_argument("--footage-date")
        parser.add_argument("--model-path")  # required by config schema
        parser.add_argument(
            "--save-per-image",
            action="store_true",
            help="Also write per-image summary CSV",
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
        result = aggregate(cfg, save_per_image=parsed_args.save_per_image)
        self.app.stdout.write(
            f"Wrote {len(result.sequences)} sequence rows to {result.sequence_csv}.\n"
        )
        if parsed_args.save_per_image:
            self.app.stdout.write(
                f"Wrote {len(result.per_image)} per-image rows to {result.per_image_csv}.\n"
            )
