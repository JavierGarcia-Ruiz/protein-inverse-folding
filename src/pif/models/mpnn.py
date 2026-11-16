"""ProteinMPNN-lite: Message Passing Neural Network for Inverse Protein Folding."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pif.data.constants import VOCAB_SIZE, PAD_IDX, MASK_IDX


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MPNNEncoderLayer(nn.Module):
    """Geometric message passing layer updating node and edge representations."""

    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Message MLP: [h_i, h_j, e_ij] -> node_dim
        self.msg_mlp = PositionWiseFeedForward(
            d_in=node_dim * 2 + edge_dim,
            d_out=node_dim,
            d_hidden=node_dim * 2,
            dropout=dropout
        )
        self.norm_msg = nn.LayerNorm(node_dim)

        # Node update MLP
        self.node_mlp = PositionWiseFeedForward(
            d_in=node_dim,
            d_out=node_dim,
            d_hidden=node_dim * 2,
            dropout=dropout
        )
        self.norm_node = nn.LayerNorm(node_dim)

        # Edge update MLP: [h_i, h_j, e_ij] -> edge_dim
        self.edge_mlp = PositionWiseFeedForward(
            d_in=node_dim * 2 + edge_dim,
            d_out=edge_dim,
            d_hidden=edge_dim * 2,
            dropout=dropout
        )
        self.norm_edge = nn.LayerNorm(edge_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (B, L, node_dim)
            edge_attr: (B, L, k, edge_dim)
            edge_index: (B, L, k)
            mask: (B, L) bool
            
        Returns:
            h_new: (B, L, node_dim)
            edge_new: (B, L, k, edge_dim)
        """
        B, L, k, _ = edge_attr.shape

        # Gather neighbor node features: h_j for each neighbor
        # Expand index to gather from h
        idx = edge_index.clamp(0, L - 1)  # (B, L, k)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, self.node_dim)
        h_expanded = h.unsqueeze(2).expand(-1, -1, k, -1)  # for h_i
        h_neighbors = torch.gather(
            h.unsqueeze(1).expand(-1, L, -1, -1),
            dim=2,
            index=idx_expanded
        )  # (B, L, k, node_dim)

        # Message computation
        msg_in = torch.cat([h_expanded, h_neighbors, edge_attr], dim=-1)  # (B, L, k, 2*node_dim + edge_dim)
        messages = self.msg_mlp(msg_in)  # (B, L, k, node_dim)

        # Edge update
        edge_new = self.norm_edge(edge_attr + self.edge_mlp(msg_in))

        # Neighbor aggregation (mean over k neighbors with neighbor validity mask)
        # Neighbor is valid if mask[b, j] is True
        mask_neighbors = torch.gather(
            mask.unsqueeze(1).expand(-1, L, -1),
            dim=2,
            index=idx
        ).unsqueeze(-1).float()  # (B, L, k, 1)

        sum_messages = torch.sum(messages * mask_neighbors, dim=2)  # (B, L, node_dim)
        count_neighbors = torch.clamp(torch.sum(mask_neighbors, dim=2), min=1.0)  # (B, L, 1)
        mean_messages = sum_messages / count_neighbors

        # Residual + Norm
        h_res = self.norm_msg(h + mean_messages)
        h_new = self.norm_node(h_res + self.node_mlp(h_res))

        # Mask out padded positions
        mask_expanded = mask.unsqueeze(-1).float()
        h_new = h_new * mask_expanded
        edge_new = edge_new * mask_expanded.unsqueeze(2)

        return h_new, edge_new


class MPNNDecoderLayer(nn.Module):
    """Autoregressive sequence decoder layer conditioning on structure and sequence."""

    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim

        # Sequence-aware message MLP: [h_i, h_j, s_j, e_ij]
        self.msg_mlp = PositionWiseFeedForward(
            d_in=node_dim * 2 + edge_dim,
            d_out=node_dim,
            d_hidden=node_dim * 2,
            dropout=dropout
        )
        self.norm_msg = nn.LayerNorm(node_dim)

        self.node_mlp = PositionWiseFeedForward(
            d_in=node_dim,
            d_out=node_dim,
            d_hidden=node_dim * 2,
            dropout=dropout
        )
        self.norm_node = nn.LayerNorm(node_dim)

    def forward(
        self,
        h_dec: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        causal_mask: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_dec: (B, L, node_dim) combined structural and sequence representation
            edge_attr: (B, L, k, edge_dim)
            edge_index: (B, L, k)
            causal_mask: (B, L, k) bool indicating if neighbor j is causally accessible to i
            mask: (B, L) bool
        """
        B, L, k, _ = edge_attr.shape

        idx = edge_index.clamp(0, L - 1)
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, self.node_dim)
        h_expanded = h_dec.unsqueeze(2).expand(-1, -1, k, -1)
        h_neighbors = torch.gather(
            h_dec.unsqueeze(1).expand(-1, L, -1, -1),
            dim=2,
            index=idx_expanded
        )

        msg_in = torch.cat([h_expanded, h_neighbors, edge_attr], dim=-1)
        messages = self.msg_mlp(msg_in)

        # Apply causal mask: only messages from preceding residues are included
        valid = (causal_mask & mask.unsqueeze(1).expand(-1, L, -1).gather(2, idx)).unsqueeze(-1).float()
        sum_messages = torch.sum(messages * valid, dim=2)
        count_valid = torch.clamp(torch.sum(valid, dim=2), min=1.0)
        mean_messages = sum_messages / count_valid

        h_res = self.norm_msg(h_dec + mean_messages)
        h_out = self.norm_node(h_res + self.node_mlp(h_res))

        mask_expanded = mask.unsqueeze(-1).float()
        return h_out * mask_expanded


class ProteinMPNN(nn.Module):
    """ProteinMPNN: Message Passing Neural Network for 3D Protein Inverse Folding."""

    def __init__(
        self,
        edge_in_dim: int = 104,
        node_dim: int = 128,
        edge_dim: int = 64,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        vocab_size: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.vocab_size = vocab_size

        # Edge embedding
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_in_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU()
        )

        # Node initial embedding (linear projection of residue index or coordination)
        self.node_embedding = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.LayerNorm(node_dim)
        )

        # Learnable positional embedding for residues
        self.pos_embedding = nn.Embedding(512, node_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            MPNNEncoderLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Sequence token embedding for decoder
        self.seq_embedding = nn.Embedding(VOCAB_SIZE + 1, node_dim, padding_idx=PAD_IDX)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            MPNNDecoderLayer(node_dim=node_dim, edge_dim=edge_dim, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, vocab_size)
        )

    def encode(
        self,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run geometric message passing encoder on 3D backbone graph.
        
        Returns:
            h: (B, L, node_dim) structural residue embeddings
            edge_h: (B, L, k, edge_dim) contextual edge embeddings
        """
        B, L, k, _ = edge_attr.shape
        device = edge_attr.device

        # Initial edge embedding
        edge_h = self.edge_embedding(edge_attr)

        # Pure SE(3)-invariant node initialization (zero initial node features; geometry flows through edges)
        h = torch.zeros((B, L, self.node_dim), device=device, dtype=edge_h.dtype)
        h = self.node_embedding(h) * mask.unsqueeze(-1).float()

        # Message passing layers
        for layer in self.encoder_layers:
            h, edge_h = layer(h, edge_h, edge_index, mask)

        return h, edge_h

    def forward(
        self,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        seq_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Autoregressive teacher-forced training forward pass.
        
        Args:
            coords: (B, L, 3, 3)
            edge_attr: (B, L, k, edge_in_dim)
            edge_index: (B, L, k)
            mask: (B, L) bool
            seq_tokens: (B, L) target amino acid sequence tokens
            
        Returns:
            logits: (B, L, vocab_size) prediction logits
        """
        B, L = mask.shape
        device = mask.device

        # Encode structural backbone
        h_enc, edge_h = self.encode(edge_attr, edge_index, mask)

        # Construct causal mask for neighbors:
        # Neighbor j is causally accessible to residue i if j < i
        idx = edge_index.clamp(0, L - 1)
        res_i = torch.arange(L, device=device).view(1, L, 1).expand(B, -1, edge_index.shape[2])
        causal_mask = idx < res_i  # (B, L, k)

        # Shift sequence tokens for teacher forcing (token at i-1 is used to predict i)
        start_tokens = torch.full((B, 1), MASK_IDX, device=device, dtype=torch.long)
        shifted_seq = torch.cat([start_tokens, seq_tokens[:, :-1]], dim=1)
        s_embed = self.seq_embedding(shifted_seq)

        # Combine structural context and sequence context
        h_dec = h_enc + s_embed

        for layer in self.decoder_layers:
            h_dec = layer(h_dec, edge_h, edge_index, causal_mask, mask)

        logits = self.head(h_dec)
        return logits

    def get_latent_embeddings(
        self,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract learned structural latent representations from the encoder."""
        h_enc, _ = self.encode(edge_attr, edge_index, mask)
        return h_enc  # (B, L, node_dim)

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Autoregressive sequence design generation conditioned on 3D backbone."""
        B, L = mask.shape
        device = mask.device

        h_enc, edge_h = self.encode(edge_attr, edge_index, mask)
        generated_tokens = torch.full((B, L), PAD_IDX, device=device, dtype=torch.long)

        idx = edge_index.clamp(0, L - 1)
        res_i = torch.arange(L, device=device).view(1, L, 1).expand(B, -1, edge_index.shape[2])
        causal_mask = idx < res_i

        # Autoregressive decoding from residue 0 to L-1
        for pos in range(L):
            # Sequence tokens up to current position
            shifted_seq = torch.cat([
                torch.full((B, 1), MASK_IDX, device=device, dtype=torch.long),
                generated_tokens[:, :-1]
            ], dim=1)
            s_embed = self.seq_embedding(shifted_seq)

            h_dec = h_enc + s_embed
            for layer in self.decoder_layers:
                h_dec = layer(h_dec, edge_h, edge_index, causal_mask, mask)

            pos_logits = self.head(h_dec[:, pos, :])  # (B, vocab_size)
            probs = F.softmax(pos_logits / max(temperature, 1e-4), dim=-1)
            sample_aa = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)

            generated_tokens[:, pos] = torch.where(mask[:, pos], sample_aa, PAD_IDX)

        return generated_tokens

