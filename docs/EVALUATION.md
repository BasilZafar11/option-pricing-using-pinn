# Evaluation and reporting

After training with the default architecture:

```sh
python benchmark.py --model_path ./checkpoints/fno_model_v1_best.pt
python evaluate.py --model_path ./checkpoints/fno_model_v1_best.pt --device cpu
```

Evaluation now inherits `Config` architecture and automatic device selection.
If architecture settings changed since training, restore them before loading.
The benchmark includes analytical, Monte Carlo and finite-difference comparisons;
the evaluation script writes plots and relative-error summaries under `results/`.

Report sample count, seed, hardware, package versions, checkpoint and parameter
ranges alongside accuracy and timing. Distinguish per-contract latency from batch
throughput. No numerical accuracy or speed result is claimed for this revision.
The time-coordinate mismatch described in LIMITATIONS.md must be resolved before
these comparisons support research conclusions.
