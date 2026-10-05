# Changelog

## Prepared revision â€” 2026-10-04

- Rewrite setup and experiment documentation around the actual FNO implementation.
- Preserve upstream attribution and license.
- Fix missing math and scipy imports in pricing/data helpers.
- Validate NumPy pricing inputs and handle expiry, zero spot and zero volatility
  without evaluating singular formulas.
- Define undefined MAPE explicitly and remove denominator bias.
- Correct short-maturity sampling and implement advertised generation arguments.
- Align evaluation defaults with central configuration and checkpoint names.
- Add deterministic numerical tests, CI and a temporary-dataset smoke check.
- Document unresolved time-coordinate inconsistencies and validation scope.
