from argparse import ArgumentParser

from cliff.command import Command

from ..config import load_config
from ..core import classify


class Classify(Command):
    """Classify existing two-stage detector labels without repeating detection."""

    def get_parser(self, prog_name):
        parser = ArgumentParser(prog=prog_name)
        parser.add_argument("--config", help="YAML configuration file.")
        parser.add_argument("--input-root")
        parser.add_argument("--output-root")
        parser.add_argument("--username")
        parser.add_argument("--camera-id")
        parser.add_argument("--footage-date")
        parser.add_argument("--model-path")
        parser.add_argument("--detector", choices=["best_27", "mdv6", "deepfaune_1.4", "best_28"])
        parser.add_argument(
            "--classifier",
            choices=["deepfaune_classifier", "4_camtrap", "2_artiodactyla", "2_carnivora"],
        )
        parser.add_argument("--models-dir")
        return parser

    def take_action(self, parsed_args):
        overrides = vars(parsed_args).copy()
        overrides["strategy"] = "two_stage"
        overrides.pop("config")
        cfg = load_config(parsed_args.config, overrides)
        classify(cfg)
