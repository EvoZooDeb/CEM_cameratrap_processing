from argparse import ArgumentParser

from cliff.command import Command

from ..config import load_config
from ..core import predict


class Predict(Command):
    """Run YOLO detection and write labels for each supported media file."""

    log = None

    def get_parser(self, prog_name):
        parser = ArgumentParser(
            prog=prog_name,
            description=(
                "Run the configured YOLO model on JPG, PNG, MP4, AVI, and MOV files in "
                "the selected camera-date input directory. YOLO-format labels, including "
                "confidence values, are written below the matching per-media results directory."
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
                "Save video frames produced during YOLO inference; required to visually "
                "validate video detections later."
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
        predict(cfg)
