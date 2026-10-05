# Data generation

Run `python data_generator_large.py --n_samples 100 --batch_size 10 --output ./data`.
The generator now honors these flags and `--seed`. At least ten samples are
required for nonempty default splits. Defaults remain 100,000 samples.

Each HDF5 file contains `params` (sigma, r, K, T), `V`, `Delta`, `Gamma`,
`S_grid`, and `t_template`. Surfaces have shape (samples, asset points, time points).
The generator uses calendar time t = template * T, with remaining maturity T - t.
The power template clusters calendar-time points near inception. Maturity
importance sampling separately favors short contracts when concentration > 1.

The 80/10/10 split is seeded. Generation allocates all three full surfaces in RAM:
100,000 x 256 x 64 x 4 bytes x 3 is about 19.7 GB before copies and split writing.
Batch size does not remove this allocation. Start small. Gamma labels are clipped
at their global 99.9th percentile; they are not unmodified analytical Gamma.
Existing datasets must be regenerated to use the corrected maturity sampling.
