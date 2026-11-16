"""Constants, amino acid vocabularies, CATH classes, and BLOSUM62 matrix."""

from typing import Dict, List, Tuple
import numpy as np

# Standard 20 amino acids
STANDARD_AAS: List[str] = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
]

AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(STANDARD_AAS)}
# Special tokens
PAD_IDX: int = len(STANDARD_AAS)      # 20
MASK_IDX: int = len(STANDARD_AAS) + 1  # 21
VOCAB_SIZE: int = len(STANDARD_AAS) + 2

IDX_TO_AA: Dict[int, str] = {i: aa for aa, i in AA_TO_IDX.items()}
IDX_TO_AA[PAD_IDX] = "<pad>"
IDX_TO_AA[MASK_IDX] = "<mask>"

# 3-letter to 1-letter mapping
THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}

# CATH Hierarchy Classes
CATH_CLASSES: Dict[int, str] = {
    1: "Mainly Alpha",
    2: "Mainly Beta",
    3: "Alpha-Beta",
    4: "Few Secondary Structures"
}

# CATH Color mapping (Okabe-Ito colorblind-friendly)
CATH_COLORS: Dict[int, str] = {
    1: "#E69F00",  # Orange (Mainly Alpha)
    2: "#56B4E9",  # Sky Blue (Mainly Beta)
    3: "#009E73",  # Bluish Green (Alpha-Beta)
    4: "#CC79A7",  # Reddish Purple (Few SS)
}

# Secondary Structure 3-class definitions
SS_CLASSES: Dict[str, str] = {
    "H": "Helix (alpha/3-10)",
    "E": "Sheet (beta-strand)",
    "C": "Coil/Loop"
}

SS_COLORS: Dict[str, str] = {
    "H": "#D55E00",  # Vermilion
    "E": "#0072B2",  # Blue
    "C": "#999999",  # Gray
}

# BLOSUM62 Substitution Matrix for 20 standard amino acids
# Row/Col order: A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
_BLOSUM62_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
_BLOSUM62_RAW = np.array([
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4]
], dtype=np.int32)

# Build standard alphabet indexed BLOSUM62 matrix
BLOSUM62_MATRIX = np.zeros((len(STANDARD_AAS), len(STANDARD_AAS)), dtype=np.int32)
for i, a1 in enumerate(STANDARD_AAS):
    idx1 = _BLOSUM62_ORDER.index(a1)
    for j, a2 in enumerate(STANDARD_AAS):
        idx2 = _BLOSUM62_ORDER.index(a2)
        BLOSUM62_MATRIX[i, j] = _BLOSUM62_RAW[idx1, idx2]


def blosum62_score(seq1: str, seq2: str) -> float:
    """Compute mean BLOSUM62 substitution score between two equal-length sequences."""
    if len(seq1) != len(seq2) or len(seq1) == 0:
        return 0.0
    scores = []
    for a1, a2 in zip(seq1, seq2):
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX:
            scores.append(BLOSUM62_MATRIX[AA_TO_IDX[a1], AA_TO_IDX[a2]])
    return float(np.mean(scores)) if scores else 0.0

