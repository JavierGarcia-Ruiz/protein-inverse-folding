"""Build CATH 4.2 domain benchmark dataset with 3D backbone coordinates."""

import os
import argparse
import pandas as pd
from pif.data.cath import build_cath_benchmark
from pif.utils.common import set_seed, load_config


def main():
    parser = argparse.ArgumentParser(description="Generate CATH 4.2 domain backbone benchmark.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file.")
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    data_cfg = cfg.get("data", {})

    n_domains = data_cfg.get("n_domains", 2200)
    seed = data_cfg.get("split_seed", 42)
    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")

    set_seed(seed)
    print(f"Building CATH 4.2 benchmark ({n_domains} domains, seed={seed})...")
    splits = build_cath_benchmark(
        n_domains=n_domains,
        seed=seed,
        raw_dir=raw_dir,
        processed_dir=processed_dir
    )

    print(f"Successfully generated CATH benchmark:")
    for split_name, split_df in splits.items():
        print(f"  - {split_name.capitalize()}: {len(split_df)} domains")
        class_counts = split_df["cath_class_name"].value_counts().to_dict()
        print(f"    Classes: {class_counts}")


if __name__ == "__main__":
    main()

