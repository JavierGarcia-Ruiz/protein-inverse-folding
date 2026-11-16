"""Evaluation metrics for protein inverse folding."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

from pif.data.constants import (
    STANDARD_AAS, IDX_TO_AA, AA_TO_IDX, PAD_IDX,
    BLOSUM62_MATRIX, CATH_CLASSES, SS_CLASSES, blosum62_score
)


def compute_sequence_recovery(
    pred_tokens: np.ndarray,
    target_tokens: np.ndarray,
    mask: np.ndarray
) -> float:
    """Compute overall sequence identity recovery percentage.
    
    Args:
        pred_tokens: (N,) predicted amino acid token indices.
        target_tokens: (N,) ground-truth amino acid token indices.
        mask: (N,) boolean valid residue mask.
        
    Returns:
        recovery_pct: Percentage (0.0 to 100.0).
    """
    valid = mask & (target_tokens != PAD_IDX)
    if not np.any(valid):
        return 0.0
    matches = (pred_tokens[valid] == target_tokens[valid])
    return float(np.mean(matches) * 100.0)


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> float:
    """Compute sequence perplexity (PPL = exp(cross_entropy)).
    
    Args:
        logits: (B, L, 20) logits over standard amino acids.
        targets: (B, L) target tokens.
        mask: (B, L) boolean valid residue mask.
        
    Returns:
        ppl: Perplexity value.
    """
    valid_mask = mask & (targets < 20) & (targets != PAD_IDX)
    if not torch.any(valid_mask):
        return 20.0

    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]

    loss = F.cross_entropy(valid_logits, valid_targets, reduction="mean")
    ppl = torch.exp(loss).item()
    return float(ppl)


def compute_inverse_folding_metrics(
    pred_tokens_list: List[np.ndarray],
    target_tokens_list: List[np.ndarray],
    sec_structs_list: List[str],
    environments_list: List[str],
    cath_classes_list: List[int],
    ppl_value: float
) -> Dict[str, Any]:
    """Compute comprehensive inverse folding benchmark metrics across structural categories."""
    all_preds = []
    all_targets = []
    all_ss = []
    all_env = []
    all_cath = []
    blosum_scores = []

    for pred_seq, targ_seq, ss, env, c_class in zip(
        pred_tokens_list, target_tokens_list, sec_structs_list, environments_list, cath_classes_list
    ):
        L = min(len(pred_seq), len(targ_seq), len(ss), len(env))
        p = pred_seq[:L]
        t = targ_seq[:L]
        all_preds.append(p)
        all_targets.append(t)
        all_ss.extend(list(ss[:L]))
        all_env.extend(list(env[:L]))
        all_cath.extend([c_class] * L)

        p_str = "".join([IDX_TO_AA.get(idx, "X") for idx in p])
        t_str = "".join([IDX_TO_AA.get(idx, "X") for idx in t])
        blosum_scores.append(blosum62_score(p_str, t_str))

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    ss_arr = np.array(all_ss)
    env_arr = np.array(all_env)
    cath_arr = np.array(all_cath)

    valid = (targets < 20) & (targets != PAD_IDX)
    overall_recovery = float(np.mean(preds[valid] == targets[valid]) * 100.0)
    mean_blosum = float(np.mean(blosum_scores)) if blosum_scores else 0.0

    # Secondary structure breakdown (Helix, Sheet, Coil)
    ss_recovery = {}
    for code, name in [("H", "helix"), ("E", "sheet"), ("C", "loop")]:
        sub_valid = valid & (ss_arr == code)
        if np.any(sub_valid):
            ss_recovery[f"recovery_{name}"] = round(float(np.mean(preds[sub_valid] == targets[sub_valid]) * 100.0), 2)
        else:
            ss_recovery[f"recovery_{name}"] = 0.0

    # Structural environment breakdown (Core, Boundary, Surface)
    env_recovery = {}
    for code, name in [("C", "core"), ("B", "boundary"), ("S", "surface")]:
        sub_valid = valid & (env_arr == code)
        if np.any(sub_valid):
            env_recovery[f"recovery_{name}"] = round(float(np.mean(preds[sub_valid] == targets[sub_valid]) * 100.0), 2)
        else:
            env_recovery[f"recovery_{name}"] = 0.0

    # CATH class breakdown
    cath_recovery = {}
    for c_id, c_name in CATH_CLASSES.items():
        sub_valid = valid & (cath_arr == c_id)
        slug = c_name.lower().replace(" ", "_").replace("-", "_")
        if np.any(sub_valid):
            cath_recovery[f"recovery_{slug}"] = round(float(np.mean(preds[sub_valid] == targets[sub_valid]) * 100.0), 2)
        else:
            cath_recovery[f"recovery_{slug}"] = 0.0

    return {
        "sequence_recovery": round(overall_recovery, 2),
        "perplexity": round(ppl_value, 2),
        "blosum62_score": round(mean_blosum, 3),
        **ss_recovery,
        **env_recovery,
        **cath_recovery
    }

