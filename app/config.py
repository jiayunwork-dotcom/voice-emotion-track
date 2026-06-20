import yaml
import os

_CONFIG = None
_METRICS_CONFIG = None

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_config():
    global _CONFIG
    if _CONFIG is None:
        config_path = os.path.join(_BASE_DIR, "config.yaml")
        _CONFIG = _load_config(config_path)
    return _CONFIG


def get_metrics_config():
    global _METRICS_CONFIG
    if _METRICS_CONFIG is None:
        metrics_path = os.path.join(_BASE_DIR, "metrics_config.yaml")
        _METRICS_CONFIG = _load_config(metrics_path)
    return _METRICS_CONFIG
