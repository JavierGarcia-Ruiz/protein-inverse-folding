"""Training and evaluation pipeline for protein inverse folding benchmarks."""

import os
import argparse
import json
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from pif.data.dataset import CATHDataset, collate_cath
from pif.models.factory import create_model
from pif.training.trainer import InverseFoldingTrainer
from pif.utils.common import set_seed, load_config, save_json


def main():
    parser = argparse.ArgumentParser(description="Train inverse protein folding models.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    set_seed(config.get("training", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} (Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load data
    raw_dir = config["data"]["raw_dir"]
    coords_path = os.path.join(raw_dir, "cath_backbone_coords.npz")
    if not os.path.exists(coords_path):
        print(f"Coordinates not found at {coords_path}. Running download_data.py first...")
        from pif.data.cath import build_cath_benchmark
        build_cath_benchmark(
            n_domains=config["data"].get("n_domains", 2200),
            seed=config["data"].get("split_seed", 42),
            raw_dir=raw_dir,
            processed_dir=config["data"]["processed_dir"]
        )

    coords_data = np.load(coords_path)
    coords_dict = {k: coords_data[k] for k in coords_data.files}

    proc_dir = config["data"]["processed_dir"]
    train_df = pd.read_csv(os.path.join(proc_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(proc_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(proc_dir, "test.csv"))

    k_neighbors = config["data"].get("num_neighbors", 30)
    batch_size = config["training"].get("batch_size", 16)

    train_ds = CATHDataset(train_df, coords_dict, k_neighbors=k_neighbors)
    val_ds = CATHDataset(val_df, coords_dict, k_neighbors=k_neighbors)
    test_ds = CATHDataset(test_df, coords_dict, k_neighbors=k_neighbors)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_cath)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_cath)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_cath)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    results = []

    # 1. Frequency Baseline
    print("\n" + "=" * 60)
    print("Evaluating Model 1: FrequencyBaseline (Empirical Background AA Null)")
    print("=" * 60)
    freq_model = create_model("frequency_baseline", config).to(device)
    freq_trainer = InverseFoldingTrainer(freq_model, train_loader, val_loader, config, device)
    _, test_ppl_f, test_rec_f, test_metrics_f = freq_trainer.evaluate(test_loader)
    test_metrics_f["model"] = "FrequencyBaseline"
    results.append(test_metrics_f)
    print(f"[FrequencyBaseline] Test Recovery: {test_rec_f:.2f}% | Test PPL: {test_ppl_f:.2f} | BLOSUM62: {test_metrics_f['blosum62_score']:.3f}")

    # 2. Sequence Transformer Baseline (Structure-blind)
    print("\n" + "=" * 60)
    print("Training Model 2: SequenceTransformer (Structure-Blind Ablation)")
    print("=" * 60)
    seq_model = create_model("seq_transformer", config).to(device)
    # Train for fewer epochs for ablation baseline
    config_seq = dict(config)
    config_seq["training"] = dict(config["training"])
    config_seq["training"]["epochs"] = min(config["training"].get("epochs", 15), 8)
    seq_trainer = InverseFoldingTrainer(seq_model, train_loader, val_loader, config_seq, device)
    seq_save_path = os.path.join(config["paths"]["models_dir"], "seq_transformer.pt")
    seq_trainer.fit(save_path=seq_save_path)
    _, test_ppl_s, test_rec_s, test_metrics_s = seq_trainer.evaluate(test_loader)
    test_metrics_s["model"] = "SequenceTransformer"
    results.append(test_metrics_s)
    print(f"[SequenceTransformer] Test Recovery: {test_rec_s:.2f}% | Test PPL: {test_ppl_s:.2f} | BLOSUM62: {test_metrics_s['blosum62_score']:.3f}")

    # 3. ProteinMPNN (3D Geometric Backbone Inverse Folding)
    print("\n" + "=" * 60)
    print("Training Model 3: ProteinMPNN (Invariant Geometric Encoder + Decoder)")
    print("=" * 60)
    mpnn_model = create_model("protein_mpnn", config).to(device)
    mpnn_trainer = InverseFoldingTrainer(mpnn_model, train_loader, val_loader, config, device)
    mpnn_save_path = os.path.join(config["paths"]["models_dir"], "protein_mpnn_best.pt")
    mpnn_trainer.fit(save_path=mpnn_save_path)

    # Load best checkpoint
    if os.path.exists(mpnn_save_path):
        ckpt = torch.load(mpnn_save_path, map_location=device)
        mpnn_model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Checkpoint] Loaded best ProteinMPNN weights from epoch {ckpt['epoch']}")

    _, test_ppl_m, test_rec_m, test_metrics_m = mpnn_trainer.evaluate(test_loader)
    test_metrics_m["model"] = "ProteinMPNN"
    results.append(test_metrics_m)
    print(f"[ProteinMPNN] Test Recovery: {test_rec_m:.2f}% | Test PPL: {test_ppl_m:.2f} | BLOSUM62: {test_metrics_m['blosum62_score']:.3f}")

    # Save benchmark table
    res_dir = config["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    res_df = pd.DataFrame(results)
    cols = ["model", "sequence_recovery", "perplexity", "blosum62_score",
            "recovery_helix", "recovery_sheet", "recovery_loop",
            "recovery_core", "recovery_boundary", "recovery_surface"]
    present_cols = [c for c in cols if c in res_df.columns]
    res_df = res_df[present_cols]

    csv_path = os.path.join(res_dir, "metrics_benchmarks.csv")
    json_path = os.path.join(res_dir, "metrics_benchmarks.json")
    res_df.to_csv(csv_path, index=False)
    save_json(results, json_path)

    print("\n" + "=" * 70)
    print("FINAL TEST BENCHMARK COMPARISON:")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print(f"\nSaved benchmark table to {csv_path}")


if __name__ == "__main__":
    main()

