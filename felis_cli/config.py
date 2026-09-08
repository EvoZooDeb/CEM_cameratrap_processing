import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_LOCATIONS = [
    Path.cwd() / ".felis.yml",
    Path.home() / ".config" / "felis" / "config.yml",
]


@dataclass
class Paths:
    input_root: Path
    output_root: Path
    username: str
    camera_id: str
    footage_date: str


@dataclass
class PredictParams:
    model_path: Path
    device: str = "cuda:0"
    imgsz: int = 1280
    conf: float = 0.25
    iou: float = 0.45
    save_frames: bool = False


@dataclass
class TwoStageParams:
    """Settings for detector-plus-classifier inference."""

    strategy: str = "two_stage"
    detector: str = "best_27"
    classifier: str = "deepfaune_classifier"
    models_dir: Path = Path("models")


@dataclass
class Config:
    paths: Paths
    predict: PredictParams
    two_stage: TwoStageParams


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def _from_env() -> Dict[str, Any]:
    return {
        "input_root": _env("FELIS_INPUT_ROOT"),
        "output_root": _env("FELIS_OUTPUT_ROOT"),
        "username": _env("FELIS_USERNAME"),
        "camera_id": _env("FELIS_CAMERA_ID"),
        "footage_date": _env("FELIS_FOOTAGE_DATE"),
        "model_path": _env("FELIS_MODEL_PATH"),
        "device": _env("FELIS_DEVICE"),
        "imgsz": _env("FELIS_IMGSZ"),
        "conf": _env("FELIS_CONF"),
        "iou": _env("FELIS_IOU"),
        "save_frames": _env("FELIS_SAVE_FRAMES"),
        "strategy": _env("FELIS_STRATEGY"),
        "detector": _env("FELIS_DETECTOR"),
        "classifier": _env("FELIS_CLASSIFIER"),
        "models_dir": _env("FELIS_MODELS_DIR"),
    }


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load configuration with precedence: CLI overrides > env > file > defaults."""
    data: Dict[str, Any] = {}

    # File
    path = config_path
    if path is None:
        env_path = _env("FELIS_CONFIG")
        if env_path:
            path = Path(env_path)
        else:
            for cand in DEFAULT_CONFIG_LOCATIONS:
                if cand.exists():
                    path = cand
                    break
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data.update(yaml.safe_load(f) or {})

    # Env
    env = {k: v for k, v in _from_env().items() if v is not None}
    data.update(env)

    # CLI overrides
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    # Required
    try:
        input_root = Path(data["input_root"]).expanduser()
        output_root = Path(data["output_root"]).expanduser()
        username = str(data["username"])  # project/survey name
        camera_id = str(data["camera_id"])  # camera unit
        footage_date = str(data["footage_date"])  # YYYYMMDD
        model_path = Path(data["model_path"]).expanduser()
    except KeyError as e:
        missing = e.args[0]
        raise SystemExit(f"Missing required config value: {missing}")

    # Optionals
    device = str(data.get("device", "cuda:0"))
    imgsz = int(data.get("imgsz", 1280))
    conf = float(data.get("conf", 0.25))
    iou = float(data.get("iou", 0.45))
    save_frames = str(data.get("save_frames", "false")).lower() in {"1", "true", "yes"}
    strategy = str(data.get("strategy", "two_stage"))
    detector = str(data.get("detector", "best_27"))
    classifier = str(data.get("classifier", "deepfaune_classifier"))
    models_dir = Path(data.get("models_dir", "models")).expanduser()

    if strategy not in {"single_stage", "two_stage"}:
        raise SystemExit("strategy must be 'single_stage' or 'two_stage'")
    detector_classifiers = {
        "best_27": {"deepfaune_classifier", "4_camtrap"},
        "mdv6": {"deepfaune_classifier", "4_camtrap"},
        "deepfaune_1.4": {"deepfaune_classifier", "4_camtrap"},
        "best_28": {"2_artiodactyla", "2_carnivora"},
    }
    if strategy == "two_stage":
        if detector not in detector_classifiers:
            raise SystemExit(f"Unknown two-stage detector: {detector}")
        if classifier not in detector_classifiers[detector]:
            raise SystemExit(
                f"Classifier '{classifier}' is not supported with detector '{detector}'"
            )

    return Config(
        paths=Paths(
            input_root=input_root,
            output_root=output_root,
            username=username,
            camera_id=camera_id,
            footage_date=footage_date,
        ),
        predict=PredictParams(
            model_path=model_path,
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save_frames=save_frames,
        ),
        two_stage=TwoStageParams(
            strategy=strategy,
            detector=detector,
            classifier=classifier,
            models_dir=models_dir,
        ),
    )
