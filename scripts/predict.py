"""CLI for autoregressive sequence design on protein 3D backbone coordinates."""

import os
import argparse
from typing import Tuple, List, Optional, Dict
import numpy as np
import torch
from Bio.PDB import PDBParser

from pif.data.constants import (
    STANDARD_AAS, AA_TO_IDX, IDX_TO_AA, THREE_TO_ONE,
    PAD_IDX, blosum62_score
)
from pif.data.features import extract_invariant_edge_features, compute_dihedrals, assign_secondary_structure
from pif.models.factory import create_model
from pif.utils.common import load_config, set_seed, format_sequence_alignment


def parse_pdb_backbone(pdb_path: str) -> Tuple[np.ndarray, str]:
    """Parse N, CA, C, O backbone coordinates and amino acid sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    coords = []
    seq_chars = []

    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            resname = res.get_resname()
            aa = THREE_TO_ONE.get(resname, "X")
            if aa not in AA_TO_IDX:
                continue

            try:
                n = res["N"].get_coord()
                ca = res["CA"].get_coord()
                c = res["C"].get_coord()
                o = res["O"].get_coord() if "O" in res else c + np.array([0, 1.2, 0])
                coords.append([n, ca, c, o])
                seq_chars.append(aa)
            except KeyError:
                continue

    return np.array(coords, dtype=np.float32), "".join(seq_chars)


def main():
    parser = argparse.ArgumentParser(description="Autoregressively design sequences for a 3D backbone.")
    parser.add_argument("--pdb", type=str, default=None, help="Path to input PDB backbone file.")
    parser.add_argument("--checkpoint", type=str, default="models/protein_mpnn_best.pt", help="Model checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (lower = more greedy).")
    parser.add_argument("--num_designs", type=int, default=3, help="Number of candidate designs to generate.")
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_model("protein_mpnn", cfg).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded ProteinMPNN checkpoint from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Running with randomly initialized weights.")

    model.eval()

    # Load or generate backbone
    if args.pdb and os.path.exists(args.pdb):
        coords_4, native_seq = parse_pdb_backbone(args.pdb)
        domain_name = os.path.basename(args.pdb)
    else:
        # Generate sample alpha-beta backbone
        from pif.data.cath import generate_ideal_backbone
        rng = np.random.RandomState(42)
        coords_4, native_seq = generate_ideal_backbone(length=100, cath_class=3, rng=rng)
        domain_name = "Sample_Alpha_Beta_Domain (L=100)"

    coords_3 = coords_4[:, :3, :]  # N, CA, C
    L = len(coords_3)
    dihedrals = compute_dihedrals(coords_4)
    ss = assign_secondary_structure(dihedrals)

    print("\n" + "=" * 70)
    print(f"PROTEIN INVERSE FOLDING SEQUENCE DESIGN: {domain_name}")
    print(f"Length: {L} residues | Temperature: {args.temperature}")
    print("=" * 70)
    print(f"Secondary Structure: {''.join(ss[:60])}...")
    print(f"Native Sequence:     {native_seq[:60]}...\n")

    # Extract invariant graph features
    coords_t = torch.from_numpy(coords_3).to(device)
    edge_attr, edge_index = extract_invariant_edge_features(
        coords_t,
        k_neighbors=cfg.get("data", {}).get("num_neighbors", 30)
    )

    # Batch tensors
    coords_b = coords_t.unsqueeze(0)
    edge_attr_b = edge_attr.unsqueeze(0)
    edge_index_b = edge_index.unsqueeze(0)
    mask_b = torch.ones((1, L), dtype=torch.bool, device=device)

    print(f"Generating {args.num_designs} sequence designs with ProteinMPNN:")
    print("-" * 70)

    for i in range(1, args.num_designs + 1):
        set_seed(42 + i * 100)
        with torch.no_grad():
            sample_tokens = model.sample(
                coords_b, edge_attr_b, edge_index_b, mask_b,
                temperature=args.temperature
            )[0].cpu().numpy()

        designed_seq = "".join([IDX_TO_AA.get(idx, "X") for idx in sample_tokens[:L]])

        # Compute recovery and BLOSUM62 score
        matches = sum(a == b for a, b in zip(designed_seq, native_seq))
        recovery_pct = (matches / L) * 100.0
        blosum = blosum62_score(designed_seq, native_seq)

        print(f"Design #{i} [Recovery: {recovery_pct:.1f}% | BLOSUM62: {blosum:+.2f}]:")
        print(f"  Seq: {designed_seq}")
        match_str = "".join(["|" if a == b else "." for a, b in zip(native_seq, designed_seq)])
        print(f"  Mch: {match_str}")
        print()


if __name__ == "__main__":
    main()

