"""PyTorch Dataset and collate function for CATH 3D backbone inverse folding."""

from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pif.data.constants import AA_TO_IDX, PAD_IDX
from pif.data.features import extract_invariant_edge_features


class CATHDataset(Dataset):
    """Dataset of protein 3D backbone coordinates and amino acid sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        coords_dict: Dict[str, np.ndarray],
        k_neighbors: int = 30,
        num_rbf: int = 16,
        max_radius: float = 20.0,
        max_len: int = 256
    ):
        self.df = df.reset_index(drop=True)
        self.coords_dict = coords_dict
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf
        self.max_radius = max_radius
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        domain_id = row["domain_id"]
        coords = self.coords_dict[domain_id][:self.max_len, :3, :]  # (L, 3, 3) for N, CA, C
        seq_str = row["sequence"][:self.max_len]
        L = len(seq_str)

        # Convert sequence to token indices
        seq_tokens = np.array([AA_TO_IDX.get(a, PAD_IDX) for a in seq_str], dtype=np.int64)

        # Invariant edge features
        coords_t = torch.from_numpy(coords)
        edge_attr, edge_index = extract_invariant_edge_features(
            coords_t,
            k_neighbors=self.k_neighbors,
            num_rbf=self.num_rbf,
            max_radius=self.max_radius
        )

        return {
            "domain_id": domain_id,
            "cath_class": int(row["cath_class"]),
            "length": L,
            "coords": coords_t,                        # (L, 3, 3)
            "seq_tokens": torch.from_numpy(seq_tokens),  # (L,)
            "edge_attr": edge_attr,                     # (L, k, edge_dim)
            "edge_index": edge_index,                   # (L, k)
            "sec_struct": row["sec_struct"][:L],
            "environments": row["environments"][:L]
        }


def collate_cath(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate variable length protein structures with padding and attention masks.
    
    Batches are padded to maximum sequence length in the batch.
    """
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    k_neighbors = batch[0]["edge_index"].shape[1]
    edge_dim = batch[0]["edge_attr"].shape[2]

    # Initialize padded tensors
    padded_coords = torch.zeros(batch_size, max_len, 3, 3, dtype=torch.float32)
    padded_seq = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padded_edge_attr = torch.zeros(batch_size, max_len, k_neighbors, edge_dim, dtype=torch.float32)
    padded_edge_index = torch.zeros(batch_size, max_len, k_neighbors, dtype=torch.long)

    cath_classes = []
    domain_ids = []
    sec_structs = []
    environments = []
    lengths = []

    for i, item in enumerate(batch):
        L = item["length"]
        lengths.append(L)
        domain_ids.append(item["domain_id"])
        cath_classes.append(item["cath_class"])
        sec_structs.append(item["sec_struct"])
        environments.append(item["environments"])

        padded_coords[i, :L] = item["coords"]
        padded_seq[i, :L] = item["seq_tokens"]
        mask[i, :L] = True
        padded_edge_attr[i, :L] = item["edge_attr"]
        padded_edge_index[i, :L] = item["edge_index"]

    return {
        "coords": padded_coords,                # (B, L_max, 3, 3)
        "seq_tokens": padded_seq,              # (B, L_max)
        "mask": mask,                          # (B, L_max) bool
        "edge_attr": padded_edge_attr,          # (B, L_max, k, edge_dim)
        "edge_index": padded_edge_index,        # (B, L_max, k)
        "cath_classes": torch.tensor(cath_classes, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "domain_ids": domain_ids,
        "sec_structs": sec_structs,
        "environments": environments
    }

