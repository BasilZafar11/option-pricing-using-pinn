# Contributing

Create a branch, describe the behavior being changed and add numerical regression
coverage when the mathematics changes. Run `python -m pytest -q` and
`python scripts/smoke_check.py` before proposing a change. Keep random seeds explicit.

For time-coordinate repairs, first define calendar time versus remaining maturity
and test the same contract at the same coordinates across generation, training,
query, Greeks and evaluation. Keep measured results separate from design goals.
Do not commit datasets, checkpoints, credentials or local environments. Preserve
the MIT notice and original contributor history.
