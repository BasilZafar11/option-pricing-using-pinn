import pytest
import torch
from utils import black_scholes_price, black_scholes_price_torch


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_tensor_price_and_delta(kind):
    s = torch.tensor(100., dtype=torch.float64, requires_grad=True)
    k, t, sigma, r = [torch.tensor(v, dtype=torch.float64) for v in (100., 1., .2, .05)]
    price = black_scholes_price_torch(s,k,t,sigma,r,kind)
    assert price.item() == pytest.approx(black_scholes_price(100,100,1,.2,.05,kind))
    price.backward()
    expected = .6368306511756191 - (kind == 'put')
    assert s.grad.item() == pytest.approx(expected)
