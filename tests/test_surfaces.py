import numpy as np
from data_generator_large import generate_batch
from utils import black_scholes_price


def test_surface_matches_independent_price_utility():
    params = np.array([[.2,.05,100.,1.]])
    spots = np.array([80.,100.,120.])
    times = np.array([0.,.5,1.])
    result = generate_batch(params, spots, times)
    expected = black_scholes_price(spots[:,None],100,1-times[None,:],.2,.05)
    np.testing.assert_allclose(result['V'][0], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result['Delta'][0,:,-1], [0,0,1])
    np.testing.assert_allclose(result['Gamma'][0,:,-1], 0)
