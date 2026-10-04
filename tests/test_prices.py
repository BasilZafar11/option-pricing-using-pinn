import numpy as np
import pytest
from utils import black_scholes_price


def test_reference_and_parity():
    call = black_scholes_price(100, 100, 1, .2, .05)
    put = black_scholes_price(100, 100, 1, .2, .05, 'put')
    assert call == pytest.approx(10.450583572185565)
    assert call - put == pytest.approx(100 - 100 * np.exp(-.05))


def test_broadcast_degenerate_inputs_without_warnings():
    with np.errstate(all='raise'):
        prices = black_scholes_price([0, 100, 110], 100, [1, 0, 1], [0, .2, 0], 0)
    np.testing.assert_allclose(prices, [0, 0, 10])


@pytest.mark.parametrize('args', [(-1,100,1,.2,0), (100,0,1,.2,0),
    (100,100,-1,.2,0), (100,100,1,-.2,0), (np.nan,100,1,.2,0)])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        black_scholes_price(*args)


def test_option_type_validation():
    with pytest.raises(ValueError):
        black_scholes_price(100,100,1,.2,0,'invalid')
