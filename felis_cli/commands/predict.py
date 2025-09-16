from argparse import ArgumentParser

from cliff.command import Command

from ..config import load_config
from ..core import predict


class Predict(Command):
    """Run YOLO detection over input files."""

    log = None

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
        parser.add_argument("--save-frames", action="store_true", help="Save frames for video inputs")
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
        predict(cfg)

