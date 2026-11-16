"""Unit tests for CATH geometric feature extraction, invariant edges, and dataset collation."""

import numpy as np
import pandas as pd
import torch

from pif.data.constants import STANDARD_AAS
from pif.data.dataset import CATHDataset, collate_cath
from pif.data.features import (
    assign_secondary_structure,
    assign_structural_environment,
    compute_dihedrals,
    extract_invariant_edge_features,
)


def _dummy_backbone_coords(length: int, seed: int = 42) -> np.ndarray:
    """Create non-degenerate backbone (L, 4, 3) test coordinates with ~3.8A CA spacing."""
    rng = np.random.RandomState(seed)
    ca = np.zeros((length, 3), dtype=np.float32)
    ca[:, 0] = np.arange(length) * 3.8
    ca[:, 1] = np.sin(np.arange(length) * 0.6) * 1.5 + rng.normal(0, 0.1, length)
    ca[:, 2] = np.cos(np.arange(length) * 0.6) * 1.5 + rng.normal(0, 0.1, length)
    coords = np.zeros((length, 4, 3), dtype=np.float32)
    coords[:, 1, :] = ca
    coords[:, 0, :] = ca + np.array([-0.9, -0.5, 0.2], dtype=np.float32)
    coords[:, 2, :] = ca + np.array([0.9, 0.5, 0.1], dtype=np.float32)
    coords[:, 3, :] = ca + np.array([1.2, 1.1, -0.3], dtype=np.float32)
    return coords


def test_features_and_dihedrals():
    coords = _dummy_backbone_coords(40)
    dihedrals = compute_dihedrals(coords)
    assert dihedrals.shape == (40, 3)

    ss = assign_secondary_structure(dihedrals)
    assert len(ss) == 40
    assert all(s in ["H", "E", "C"] for s in ss)

    envs = assign_structural_environment(coords[:, 1, :])
    assert len(envs) == 40
    assert all(e in ["Core", "Boundary", "Surface"] for e in envs)


def test_invariant_edge_features():
    coords = torch.randn(30, 3, 3)
    edge_attr, edge_index = extract_invariant_edge_features(coords, k_neighbors=10)
    assert edge_attr.shape == (30, 10, 104)
    assert edge_index.shape == (30, 10)
    assert not torch.isnan(edge_attr).any()


def test_cath_dataset_and_collate():
    coords1 = _dummy_backbone_coords(30, seed=1)
    coords2 = _dummy_backbone_coords(45, seed=2)
    seq1 = "".join(STANDARD_AAS[i % len(STANDARD_AAS)] for i in range(30))
    seq2 = "".join(STANDARD_AAS[i % len(STANDARD_AAS)] for i in range(45))

    coords_dict = {"d1": coords1, "d2": coords2}
    df = pd.DataFrame([
        {"domain_id": "d1", "cath_class": 1, "length": 30, "sequence": seq1, "sec_struct": "H" * 30, "environments": "C" * 30},
        {"domain_id": "d2", "cath_class": 2, "length": 45, "sequence": seq2, "sec_struct": "E" * 45, "environments": "S" * 45},
    ])

    ds = CATHDataset(df, coords_dict, k_neighbors=15)
    item = ds[0]
    assert item["coords"].shape == (30, 3, 3)
    assert item["edge_attr"].shape == (30, 15, 104)

    batch = collate_cath([ds[0], ds[1]])
    assert batch["coords"].shape == (2, 45, 3, 3)
    assert batch["mask"].shape == (2, 45)
    assert batch["mask"][0, 30:].sum() == 0
    assert batch["mask"][1, :45].sum() == 45
