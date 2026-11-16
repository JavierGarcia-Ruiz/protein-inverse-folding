"""Model factory for inverse protein folding architectures."""

from typing import Dict, Any
import torch.nn as nn

from pif.models.baselines import FrequencyBaseline, SequenceTransformerBaseline
from pif.models.mpnn import ProteinMPNN


def create_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """Factory function to instantiate inverse folding models.
    
    Supported models:
        - 'frequency_baseline': Background amino acid frequency null model.
        - 'seq_transformer': 1D Autoregressive sequence-only Transformer (structure ablation).
        - 'protein_mpnn': Invariant Message Passing Neural Network on 3D backbone.
    """
    model_cfg = config.get("model", {})
    name = model_name.lower().replace("-", "_")

    if name in ["frequency_baseline", "frequency"]:
        return FrequencyBaseline(vocab_size=model_cfg.get("vocab_size", 20))

    elif name in ["seq_transformer", "sequence_transformer", "transformer"]:
        return SequenceTransformerBaseline(
            hidden_dim=model_cfg.get("node_dim", 128),
            num_layers=model_cfg.get("num_decoder_layers", 3),
            num_heads=4,
            vocab_size=model_cfg.get("vocab_size", 20),
            max_len=config.get("data", {}).get("max_len", 256),
            dropout=model_cfg.get("dropout", 0.1)
        )

    elif name in ["protein_mpnn", "mpnn", "proteinmpnn"]:
        return ProteinMPNN(
            edge_in_dim=104,
            node_dim=model_cfg.get("node_dim", 128),
            edge_dim=model_cfg.get("edge_dim", 64),
            num_encoder_layers=model_cfg.get("num_encoder_layers", 3),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 3),
            vocab_size=model_cfg.get("vocab_size", 20),
            dropout=model_cfg.get("dropout", 0.1)
        )

    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from 'frequency_baseline', 'seq_transformer', 'protein_mpnn'.")

