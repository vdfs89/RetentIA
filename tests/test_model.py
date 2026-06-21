import torch

from src.models.mlp import ChurnMLP


def test_mlp_forward_shape():
    input_dim = 20
    model = ChurnMLP(input_dim)
    model.eval()
    x = torch.randn(5, input_dim)
    out = model(x)
    assert out.shape == (5, 1)


def test_mlp_sigmoid_range():
    model = ChurnMLP(input_dim=20)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.randn(8, 20)))
    assert torch.all(prob >= 0) and torch.all(prob <= 1)
