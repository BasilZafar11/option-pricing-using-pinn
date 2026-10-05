import pytest
from utils import black_scholes_price, monte_carlo_price


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_seeded_estimate_agrees_with_analytical_reference(kind):
    args = (100, 100, 1, .2, .05)
    price, error = monte_carlo_price(*args, n_paths=100000, seed=42, option_type=kind)
    assert error > 0
    assert abs(price - black_scholes_price(*args, option_type=kind)) < 5 * error
    assert (price, error) == monte_carlo_price(*args, n_paths=100000, seed=42, option_type=kind)
