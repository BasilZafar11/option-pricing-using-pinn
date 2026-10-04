import numpy as np
import pytest
from data_generator_large import importance_sample_T, latin_hypercube_sample


def test_maturities_concentrate_near_lower_bound():
    sample = importance_sample_T(10000, 0, 1, concentration=2, seed=42)
    assert sample.mean() < .4
    np.testing.assert_array_equal(sample, importance_sample_T(10000,0,1,2,42))


@pytest.mark.parametrize('value', [0, -1, float('nan')])
def test_invalid_concentration(value):
    with pytest.raises(ValueError):
        importance_sample_T(10,0,1,value)


def test_latin_hypercube_strata():
    sample = latin_hypercube_sample(20, [(0,1)], seed=7)[:,0]
    assert sorted((sample*20).astype(int)) == list(range(20))
