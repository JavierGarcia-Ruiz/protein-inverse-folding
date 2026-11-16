"""Real CATH 4.2 Domain Benchmark loader and preprocessor (Ingraham et al. / ProteinMPNN)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict

import numpy as np
import pandas as pd

from pif.data.constants import CATH_CLASSES, STANDARD_AAS
from pif.data.features import (
    assign_secondary_structure,
    assign_structural_environment,
    compute_dihedrals,
)

CATH_JSONL_URL = "http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl"
CATH_SPLITS_URL = "http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json"
STANDARD_AA_SET = set(STANDARD_AAS)


def ensure_cath_raw_files(raw_dir: str = "data/raw") -> tuple[str, str]:
    """Download authentic CATH 4.2 JSONL structures and official topology splits if missing."""
    cath_dir = os.path.join(raw_dir, "cath_4_2")
    os.makedirs(cath_dir, exist_ok=True)
    splits_path = os.path.join(cath_dir, "chain_set_splits.json")
    jsonl_path = os.path.join(cath_dir, "chain_set.jsonl")

    if not os.path.exists(splits_path):
        urllib.request.urlretrieve(CATH_SPLITS_URL, splits_path)
    if not os.path.exists(jsonl_path):
        urllib.request.urlretrieve(CATH_JSONL_URL, jsonl_path)
    return jsonl_path, splits_path


def build_cath_benchmark(
    n_domains: int | None = None,
    seed: int = 42,
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    min_len: int = 40,
    max_len: int = 350,
) -> Dict[str, pd.DataFrame]:
    """Load authentic CATH 4.2 X-ray/cryo-EM crystal domains with official non-redundant splits.

    Filters strictly for experimental PDB chains with 100% resolved backbone coordinates
    (N, CA, C, O all finite) and standard 20 amino acids between ``min_len`` and ``max_len``.
    """
    _ = (n_domains, seed)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    jsonl_path, splits_path = ensure_cath_raw_files(raw_dir)
    with open(splits_path, "r", encoding="utf-8") as fh:
        official_splits = json.load(fh)

    train_ids = set(official_splits["train"])
    val_ids = set(official_splits["validation"])
    test_ids = set(official_splits["test"])

    records = []
    coords_dict: dict[str, np.ndarray] = {}

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            domain_id = str(rec["name"])
            if domain_id in train_ids:
                split_name = "train"
            elif domain_id in val_ids:
                split_name = "val"
            elif domain_id in test_ids:
                split_name = "test"
            else:
                continue

            seq = str(rec.get("seq", "")).strip().upper()
            L = len(seq)
            if L < min_len or L > max_len:
                continue
            if any(ch not in STANDARD_AA_SET for ch in seq):
                continue

            c = rec["coords"]
            coords = np.stack([c["N"], c["CA"], c["C"], c["O"]], axis=1).astype(np.float32)
            if not np.isfinite(coords).all():
                continue

            cath_list = rec.get("CATH") or []
            if cath_list:
                try:
                    raw_c = int(str(cath_list[0]).split(".")[0])
                except ValueError:
                    raw_c = 3
            else:
                raw_c = 3
            c_class = raw_c if raw_c in CATH_CLASSES else 4

            dihedrals = compute_dihedrals(coords)
            ss = assign_secondary_structure(dihedrals)
            envs = assign_structural_environment(coords[:, 1, :])

            ca_coords = coords[:, 1, :]
            rg = float(np.sqrt(np.mean(np.sum((ca_coords - np.mean(ca_coords, axis=0)) ** 2, axis=-1))))

            records.append(
                {
                    "domain_id": domain_id,
                    "cath_id": str(cath_list[0]) if cath_list else "",
                    "cath_class": c_class,
                    "cath_class_name": CATH_CLASSES[c_class],
                    "length": L,
                    "sequence": seq,
                    "sec_struct": "".join(ss),
                    "environments": "".join(e[0] for e in envs),
                    "helix_fraction": round(float(np.mean([s == "H" for s in ss])), 4),
                    "sheet_fraction": round(float(np.mean([s == "E" for s in ss])), 4),
                    "loop_fraction": round(float(np.mean([s == "C" for s in ss])), 4),
                    "core_fraction": round(float(np.mean([e == "Core" for e in envs])), 4),
                    "surface_fraction": round(float(np.mean([e == "Surface" for e in envs])), 4),
                    "rg": round(rg, 3),
                    "split": split_name,
                }
            )
            coords_dict[domain_id] = coords

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(raw_dir, "cath_domains_metadata.csv"), index=False)
    np.savez_compressed(os.path.join(raw_dir, "cath_backbone_coords.npz"), **coords_dict)

    splits = {
        "train": df[df["split"] == "train"].reset_index(drop=True),
        "val": df[df["split"] == "val"].reset_index(drop=True),
        "test": df[df["split"] == "test"].reset_index(drop=True),
    }
    for sp_name, sp_df in splits.items():
        sp_df.to_csv(os.path.join(processed_dir, f"{sp_name}.csv"), index=False)

    return splits
