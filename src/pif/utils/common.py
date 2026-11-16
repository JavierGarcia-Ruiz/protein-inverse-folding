"""Common utility functions for seed setting, configuration, and logging."""

import os
import random
import json
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict, path: str) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def format_sequence_alignment(seq1: str, seq2: str, max_chars_per_line: int = 60) -> str:
    """Format pairwise alignment string between native and designed sequences."""
    lines = []
    for i in range(0, len(seq1), max_chars_per_line):
        s1 = seq1[i:i + max_chars_per_line]
        s2 = seq2[i:i + max_chars_per_line]
        match_line = "".join(["|" if a == b else ("." if a != b else " ") for a, b in zip(s1, s2)])
        lines.append(f"Native:   {s1}")
        lines.append(f"Match:    {match_line}")
        lines.append(f"Designed: {s2}")
        lines.append("")
    return "\n".join(lines)

