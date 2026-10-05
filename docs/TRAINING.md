# Training experiments

Generate data first, then edit `Config` in `config.py` and run `python train.py`.
Set a descriptive `run_name`. The default model has 24 Fourier modes, four
Fourier layers and width 64 on a 256 x 64 grid. Training defaults to 200 epochs.

For an exploratory local run, lower sample count and epochs, disable
`run_benchmark`, keep `num_workers = 0`, and leave cloud/W&B settings disabled.
Lowering the grid also requires compatible Fourier modes. This is not a promise
that the full default model fits a particular device.

Checkpoints are named `<run_name>_best.pt`, `<run_name>_checkpoint_latest.pt`,
and `<run_name>_final.pt`. Set `resume_from_checkpoint` to resume a compatible run.
Load only trusted checkpoints. Preserve the exact configuration with results;
evaluation must use the same architecture. Read LIMITATIONS.md before interpreting
training losses as evidence of pricing accuracy.
