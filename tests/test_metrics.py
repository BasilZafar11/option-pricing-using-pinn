import pytest
import torch
from utils import compute_mape, compute_rmse, compute_max_error


def test_metrics():
    pred, true = torch.tensor([2., 4.]), torch.tensor([1., 2.])
    assert compute_mape(pred,true).item() == pytest.approx(100.)
    assert compute_rmse(pred,true).item() == pytest.approx(2.5 ** .5)
    assert compute_max_error(pred,true).item() == 2


def test_mape_excludes_zeros_and_preserves_signed_denominator():
    assert compute_mape(torch.tensor([99., -2.]), torch.tensor([0., -1.])).item() == 100
    assert torch.isnan(compute_mape(torch.ones(2), torch.zeros(2)))
