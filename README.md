# Option Surface Lab

**Physics-informed Fourier neural operators for European option-pricing research.**

This project explores learning an entire Black–Scholes call-price surface from
volatility, interest rate, strike and maturity. It combines synthetic analytical
data, a Fourier neural operator (FNO), a coordinate-aware decoder and physics-based
training losses. The repository name retains PINN terminology; the implemented
network is an FNO with physics-informed training, rather than a conventional
pointwise PINN.

This is a continuation of the collaborative project published by
[Diclo-fenac](https://github.com/Diclo-fenac/option-pricing-using-pinn).
See [provenance](ATTRIBUTION.md) and the preserved [MIT license](LICENSE).

## What is included

- Analytical call/put utilities and Monte Carlo reference pricing.
- Synthetic call-price surfaces with Delta and clipped Gamma labels.
- Fourier layers, coordinate decoding and an expiry-payoff constraint.
- Training, checkpointing, optional cloud uploads and optional W&B logging.
- Benchmark/evaluation scripts and three exploratory notebooks.
- Numerical regression tests and a small dataset smoke check.

**Research status:** the neural pipeline has a known time-coordinate inconsistency
across generation, training and evaluation. Read [limitations](docs/LIMITATIONS.md)
before using model results. This revision makes no measured accuracy or latency claim.

## Start locally

Python 3.11+ is recommended. From the repository folder:

```sh
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` on PowerShell, or
`source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python examples/analytical_price.py
python scripts/smoke_check.py
```

The analytical example prints approximately 10.450584 for the call and 5.573526
for the put at S = K = 100, T = 1, sigma = 0.2, r = 0.05. The smoke check generates
20 small surfaces in a temporary directory and removes them afterwards.
The full requirements include optional cloud/logging packages; CI installs only
the dependencies needed for the local checks.

## Run experiments

```sh
python data_generator_large.py --n_samples 100 --batch_size 10 --output ./data
python train.py
python benchmark.py --model_path ./checkpoints/fno_model_v1_best.pt
python evaluate.py --model_path ./checkpoints/fno_model_v1_best.pt --device cpu
```

Before training, edit `config.py` for the intended budget. The example dataset is
small, but the default training remains 200 epochs. The default 100,000-sample
generation uses about 19.7 GB just for its three surface arrays, before copies.
No pretrained checkpoint is included.

## Repository map

| File | Purpose |
| --- | --- |
| `config.py` | Experiment, architecture and storage settings |
| `data_generator_large.py` | Sampling, analytical surfaces and HDF5 splits |
| `fno_model.py` | Fourier operator, decoder and physics/Greek helpers |
| `train.py` | Optimization and checkpoints |
| `utils.py` | Analytical pricing, metrics and optional storage helper |
| `fdm_solver.py` | Finite-difference reference solver |
| `benchmark.py`, `evaluate.py` | Comparisons and plots |
| `notebooks/` | Original interactive workflows |
| `tests/`, `scripts/` | Regression checks and smoke verification |

## Mathematical target

For a European call without dividends, the calendar-time PDE is

```text
dV/dt + 0.5 * sigma^2 * S^2 * d²V/dS² + r*S*dV/dS - r*V = 0
V(S, T) = max(S - K, 0)
```

The intended operator maps `(sigma, r, K, T)` to a grid of prices. Utility functions
accept remaining maturity, while dataset templates represent calendar time.
Keeping these conventions consistent is essential to a valid experiment.

## Further reading

[Data format](docs/DATA.md) · [Training](docs/TRAINING.md) ·
[Evaluation](docs/EVALUATION.md) · [Known limitations](docs/LIMITATIONS.md) ·
[Contributing](CONTRIBUTING.md) · [Changes](CHANGELOG.md)
