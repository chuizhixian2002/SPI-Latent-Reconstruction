import argparse
import os

import torch
import yaml


class DotDict(dict):
    """
    Dictionary with dot-style access.

    Example:
        cfg.train.batch_size
    """

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

        if isinstance(value, dict):
            value = DotDict(value)
            self[key] = value

        return value


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return DotDict(cfg)


def get_device(device_cfg="auto"):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_cfg)


def parse_config_arg(default_config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config, help="Path to YAML config.")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    return args
