"""Invariant 3D backbone geometric feature extraction for ProteinMPNN."""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


def compute_dihedrals(coords: np.ndarray) -> np.ndarray:
    """Compute backbone dihedral angles (phi, psi, omega) from N, CA, C coordinates.
    
    Args:
        coords: (L, 3, 3) or (L, 4, 3) coordinates for [N, CA, C, (O)]
        
    Returns:
        dihedrals: (L, 3) angles in radians [phi, psi, omega]
    """
    L = coords.shape[0]
    dihedrals = np.zeros((L, 3), dtype=np.float32)
    if L < 3:
        return dihedrals

    N = coords[:, 0, :]
    CA = coords[:, 1, :]
    C = coords[:, 2, :]

    def dihedral(p0, p1, p2, p3):
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
        b1_hat = b1 / b1_norm
        v = b0 - np.sum(b0 * b1_hat, axis=-1, keepdims=True) * b1_hat
        w = b2 - np.sum(b2 * b1_hat, axis=-1, keepdims=True) * b1_hat
        x = np.sum(v * w, axis=-1)
        y = np.sum(np.cross(b1_hat, v) * w, axis=-1)
        return np.arctan2(y, x)

    # phi: C(i-1) - N(i) - CA(i) - C(i)
    if L > 1:
        dihedrals[1:, 0] = dihedral(C[:-1], N[1:], CA[1:], C[1:])
    # psi: N(i) - CA(i) - C(i) - N(i+1)
    if L > 1:
        dihedrals[:-1, 1] = dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
    # omega: CA(i-1) - C(i-1) - N(i) - CA(i)
    if L > 1:
        dihedrals[1:, 2] = dihedral(CA[:-1], C[:-1], N[1:], CA[1:])

    return dihedrals


def assign_secondary_structure(dihedrals: np.ndarray) -> list:
    """Assign 3-state secondary structure (H=Helix, E=Sheet, C=Coil) via Ramachandran basins.
    
    Args:
        dihedrals: (L, 3) phi, psi, omega in radians.
        
    Returns:
        List of 'H', 'E', 'C' strings of length L.
    """
    phi_deg = np.degrees(dihedrals[:, 0])
    psi_deg = np.degrees(dihedrals[:, 1])
    ss = []
    for p, s in zip(phi_deg, psi_deg):
        # Alpha-helix basin
        if -160 <= p <= -30 and -80 <= s <= 10:
            ss.append("H")
        # Beta-strand basin
        elif -180 <= p <= -40 and (s >= 70 or s <= -170):
            ss.append("E")
        else:
            ss.append("C")
    return ss


def assign_structural_environment(ca_coords: np.ndarray, radius: float = 10.0) -> list:
    """Classify residues as Core, Boundary, or Surface based on CA neighbor density.
    
    Args:
        ca_coords: (L, 3) CA coordinates.
        radius: Sphere cutoff in Angstroms.
        
    Returns:
        List of 'Core', 'Boundary', 'Surface' strings.
    """
    L = len(ca_coords)
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    # Count neighbors excluding self
    neighbor_counts = np.sum((dist < radius) & (dist > 1e-4), axis=1)
    envs = []
    for count in neighbor_counts:
        if count >= 18:
            envs.append("Core")
        elif count >= 12:
            envs.append("Boundary")
        else:
            envs.append("Surface")
    return envs


def rbf_expansion(distances: torch.Tensor, num_rbf: int = 16, max_radius: float = 20.0) -> torch.Tensor:
    """Expand scalar distances into Gaussian Radial Basis Functions.
    
    Args:
        distances: (...,) tensor of pairwise distances in Angstroms.
        num_rbf: Number of Gaussian centers.
        max_radius: Maximum distance cutoff.
        
    Returns:
        (..., num_rbf) Gaussian RBF activations.
    """
    centers = torch.linspace(0.0, max_radius, num_rbf, device=distances.device, dtype=distances.dtype)
    gamma = 1.0 / ((centers[1] - centers[0]) ** 2)
    diff = distances.unsqueeze(-1) - centers
    return torch.exp(-gamma * (diff ** 2))


def build_protein_backbone_frames(coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct orthonormal local coordinate frames at each residue CA.
    
    Args:
        coords: (B, L, 3, 3) or (L, 3, 3) tensor for [N, CA, C] atoms.
        
    Returns:
        rotations: (..., L, 3, 3) orthonormal rotation matrices [b, c, a].
        translations: (..., L, 3) CA positions.
    """
    N = coords[..., 0, :]
    CA = coords[..., 1, :]
    C = coords[..., 2, :]

    # b = normalized(C - CA)
    v1 = C - CA
    b = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-7)

    # c = normalized(orthogonalized(N - CA))
    v2 = N - CA
    proj = torch.sum(v2 * b, dim=-1, keepdim=True) * b
    u = v2 - proj
    c = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-7)

    # a = b x c
    a = torch.cross(b, c, dim=-1)

    # Frame rotation R = [b, c, a]
    rot = torch.stack([b, c, a], dim=-1)
    return rot, CA


def extract_invariant_edge_features(
    coords: torch.Tensor,
    k_neighbors: int = 30,
    num_rbf: int = 16,
    max_radius: float = 20.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute invariant edge features and k-NN graph for ProteinMPNN.
    
    Features per edge (i, j):
        - CA-CA, N-N, C-C, N-CA, CA-C, N-C RBF distances (6 * num_rbf)
        - Local frame transformed direction vector: R_i^T (CA_j - CA_i) / dist (3)
        - Relative quaternion orientation: q(R_i^T R_j) (4)
        - Positional sequence separation: clipped signed embedding (1)
        
    Returns:
        edge_attr: (L, k_neighbors, edge_feat_dim)
        edge_index: (L, k_neighbors) neighbor indices
    """
    L = coords.shape[0]
    device = coords.device
    dtype = coords.dtype

    CA = coords[:, 1, :]  # (L, 3)
    N = coords[:, 0, :]   # (L, 3)
    C = coords[:, 2, :]   # (L, 3)

    # Pairwise CA-CA distances
    diff_ca = CA.unsqueeze(0) - CA.unsqueeze(1)  # (L, L, 3): CA_j - CA_i
    dist_ca = torch.norm(diff_ca, dim=-1)         # (L, L)

    # k-nearest neighbors
    k = min(k_neighbors, L)
    _, knn_idx = torch.topk(dist_ca, k=k, dim=-1, largest=False)  # (L, k)

    # Gather neighbor coordinates
    idx_flat = knn_idx.reshape(-1)
    ca_j = CA[idx_flat].view(L, k, 3)
    n_j = N[idx_flat].view(L, k, 3)
    c_j = C[idx_flat].view(L, k, 3)

    ca_i = CA.unsqueeze(1).expand(-1, k, -1)
    n_i = N.unsqueeze(1).expand(-1, k, -1)
    c_i = C.unsqueeze(1).expand(-1, k, -1)

    # Multi-atom pairwise distances
    d_ca_ca = torch.norm(ca_j - ca_i, dim=-1)
    d_n_n = torch.norm(n_j - n_i, dim=-1)
    d_c_c = torch.norm(c_j - c_i, dim=-1)
    d_n_ca = torch.norm(n_j - ca_i, dim=-1)
    d_ca_c = torch.norm(ca_j - c_i, dim=-1)
    d_n_c = torch.norm(n_j - c_i, dim=-1)

    rbf_ca_ca = rbf_expansion(d_ca_ca, num_rbf=num_rbf, max_radius=max_radius)
    rbf_n_n = rbf_expansion(d_n_n, num_rbf=num_rbf, max_radius=max_radius)
    rbf_c_c = rbf_expansion(d_c_c, num_rbf=num_rbf, max_radius=max_radius)
    rbf_n_ca = rbf_expansion(d_n_ca, num_rbf=num_rbf, max_radius=max_radius)
    rbf_ca_c = rbf_expansion(d_ca_c, num_rbf=num_rbf, max_radius=max_radius)
    rbf_n_c = rbf_expansion(d_n_c, num_rbf=num_rbf, max_radius=max_radius)

    # Local frames
    rot, _ = build_protein_backbone_frames(coords)  # (L, 3, 3)
    rot_i = rot.unsqueeze(1).expand(-1, k, -1, -1)  # (L, k, 3, 3)
    rot_j = rot[idx_flat].view(L, k, 3, 3)

    # Direction in local frame: R_i^T (ca_j - ca_i)
    delta = (ca_j - ca_i) / (d_ca_ca.unsqueeze(-1) + 1e-7)  # (L, k, 3)
    local_dir = torch.sum(rot_i * delta.unsqueeze(-1), dim=-2)  # (L, k, 3)

    # Relative rotation diagonal: diag(R_i^T R_j)
    rel_rot_diag = torch.sum(rot_i * rot_j, dim=-2)  # (L, k, 3)

    # Relative sequence separation
    seq_i = torch.arange(L, device=device).unsqueeze(1).expand(-1, k)
    seq_j = knn_idx
    seq_diff = (seq_j - seq_i).float()
    seq_sign = torch.sign(seq_diff).unsqueeze(-1)
    seq_dist_norm = (torch.clamp(torch.abs(seq_diff), max=32.0) / 32.0).unsqueeze(-1)

    # Concatenate edge features
    edge_attr = torch.cat([
        rbf_ca_ca, rbf_n_n, rbf_c_c,
        rbf_n_ca, rbf_ca_c, rbf_n_c,
        local_dir, rel_rot_diag,
        seq_sign, seq_dist_norm
    ], dim=-1)  # total: 6 * 16 + 3 + 3 + 2 = 104 dims

    return edge_attr, knn_idx
