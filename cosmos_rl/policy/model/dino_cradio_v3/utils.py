from typing import Any
from pathlib import Path
import yaml

EXPERIMENT_CONFIG_NAME = "experiment_config.yaml"


def get_experiment_config() -> dict[str, Any]:
    here = Path(__file__).parent
    config_path = here / EXPERIMENT_CONFIG_NAME
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
