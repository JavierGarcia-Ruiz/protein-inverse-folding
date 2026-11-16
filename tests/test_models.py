"""Unit tests for inverse folding models."""

import torch
import pytest

from pif.models.factory import create_model


def test_frequency_baseline():
    cfg = {"model": {"vocab_size": 20}}
    model = create_model("frequency_baseline", cfg)

    coords = torch.randn(2, 20, 3, 3)
    edge_attr = torch.randn(2, 20, 10, 104)
    edge_index = torch.zeros(2, 20, 10, dtype=torch.long)
    mask = torch.ones(2, 20, dtype=torch.bool)

    logits = model(coords, edge_attr, edge_index, mask)
    assert logits.shape == (2, 20, 20)

    samples = model.sample(coords, edge_attr, edge_index, mask)
    assert samples.shape == (2, 20)
    assert (samples >= 0).all() and (samples < 20).all()


def test_seq_transformer_baseline():
    cfg = {
        "model": {"node_dim": 64, "num_decoder_layers": 2, "vocab_size": 20, "dropout": 0.0},
        "data": {"max_len": 50}
    }
    model = create_model("seq_transformer", cfg)

    coords = torch.randn(2, 15, 3, 3)
    edge_attr = torch.randn(2, 15, 5, 104)
    edge_index = torch.zeros(2, 15, 5, dtype=torch.long)
    mask = torch.ones(2, 15, dtype=torch.bool)
    seq_tokens = torch.randint(0, 20, (2, 15))

    logits = model(coords, edge_attr, edge_index, mask, seq_tokens)
    assert logits.shape == (2, 15, 20)


def test_protein_mpnn():
    cfg = {
        "model": {
            "node_dim": 64,
            "edge_dim": 32,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "vocab_size": 20,
            "dropout": 0.0
        }
    }
    model = create_model("protein_mpnn", cfg)

    coords = torch.randn(2, 25, 3, 3)
    edge_attr = torch.randn(2, 25, 10, 104)
    edge_index = torch.randint(0, 25, (2, 25, 10))
    mask = torch.ones(2, 25, dtype=torch.bool)
    seq_tokens = torch.randint(0, 20, (2, 25))

    # Forward pass
    logits = model(coords, edge_attr, edge_index, mask, seq_tokens)
    assert logits.shape == (2, 25, 20)

    # Latent embeddings
    latents = model.get_latent_embeddings(coords, edge_attr, edge_index, mask)
    assert latents.shape == (2, 25, 64)

    # Sampling
    samples = model.sample(coords, edge_attr, edge_index, mask, temperature=0.5)
    assert samples.shape == (2, 25)
    assert (samples >= 0).all() and (samples < 20).all()

