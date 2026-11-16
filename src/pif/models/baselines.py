"""Baseline models for protein inverse folding benchmark."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pif.data.constants import VOCAB_SIZE, STANDARD_AAS, AA_TO_IDX, PAD_IDX


class FrequencyBaseline(nn.Module):
    """Null model predicting empirical amino acid background frequencies."""

    def __init__(self, vocab_size: int = 20):
        super().__init__()
        self.vocab_size = vocab_size
        # Natural amino acid background frequencies
        freqs = torch.tensor([
            0.0825, 0.0137, 0.0545, 0.0675, 0.0386,  # A, C, D, E, F
            0.0707, 0.0227, 0.0596, 0.0584, 0.0966,  # G, H, I, K, L
            0.0242, 0.0406, 0.0470, 0.0393, 0.0553,  # M, N, P, Q, R
            0.0656, 0.0534, 0.0687, 0.0108, 0.0292   # S, T, V, W, Y
        ], dtype=torch.float32)
        logits = torch.log(freqs + 1e-8)
        self.register_buffer("log_probs", logits)

    def forward(
        self,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        seq_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, L = mask.shape
        logits = self.log_probs.view(1, 1, -1).expand(B, L, -1)
        return logits

    def sample(
        self,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        logits = self.forward(coords, edge_attr, edge_index, mask)
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        B, L, V = probs.shape
        samples = torch.multinomial(probs.view(-1, V), num_samples=1).view(B, L)
        samples = torch.where(mask, samples, torch.full_like(samples, PAD_IDX))
        return samples


class SequenceTransformerBaseline(nn.Module):
    """Autoregressive sequence-only Transformer (structure-blind ablation)."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        vocab_size: int = 20,
        max_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(VOCAB_SIZE + 1, hidden_dim, padding_idx=PAD_IDX)
        self.pos_embed = nn.Embedding(max_len + 1, hidden_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        seq_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Causal mask for autoregressive teacher forcing
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device).bool()

        # Input shifted sequence (prepended with start token represented by MASK_IDX)
        start_tokens = torch.full((B, 1), PAD_IDX, device=device, dtype=torch.long)
        shifted_tokens = torch.cat([start_tokens, seq_tokens[:, :-1]], dim=1)

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(shifted_tokens) + self.pos_embed(positions)

        padding_mask = ~mask
        h_out = self.decoder(h, mask=causal_mask, is_causal=True, src_key_padding_mask=padding_mask)
        h_norm = self.norm(h_out)
        logits = self.head(h_norm)
        return logits

    def sample(
        self,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        B, L = mask.shape
        device = mask.device
        generated = torch.zeros(B, L, dtype=torch.long, device=device)

        for pos in range(L):
            logits = self.forward(coords, edge_attr, edge_index, mask, generated)
            step_logits = logits[:, pos, :] / max(temperature, 1e-4)
            probs = F.softmax(step_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated[:, pos] = next_token

        return torch.where(mask, generated, torch.full_like(generated, PAD_IDX))

