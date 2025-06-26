import torch
import pytest

from minerva.model import Decoder


@pytest.fixture
def model_config():
    """Provides a sample model configuration."""
    return {
        "vocab_size": 1000,
        "num_layers": 2,
        "hidden_dim": 128,
        "num_heads": 4,
        "max_seq_len": 256,
        "dropout": 0.1,
    }


@pytest.fixture
def dummy_input():
    """Provides a dummy input tensor."""
    return torch.randint(0, 1000, (2, 32))  # (batch_size, seq_len)


def test_decoder_forward_pass(model_config, dummy_input):
    """
    Tests that the Decoder model can perform a forward pass without errors
    and that the output shape is correct.
    """
    model = Decoder(**model_config)
    output = model(dummy_input)

    assert output is not None
    assert isinstance(output, torch.Tensor)

    expected_shape = (
        dummy_input.shape[0],
        dummy_input.shape[1],
        model_config["vocab_size"],
    )
    assert output.shape == expected_shape, (
        f"Output shape mismatch: expected {expected_shape}, got {output.shape}"
    )

def test_decoder_weight_tying(model_config):
    """
    Tests that the token embedding and the final linear layer (LM head) share weights.
    """
    model = Decoder(**model_config)
    assert model.token_emb.weight is model.lm_head.weight, "Token embedding and LM head weights are not tied" 