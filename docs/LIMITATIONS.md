# Known numerical limitations

The original neural pipeline uses inconsistent time coordinates. Generated labels
use t = template * each contract maturity. The trainer builds t = template * T_max,
and benchmark paths pass the unscaled template to a model that treats it as
absolute time. Contracts with different maturities therefore need a coordinated
change across forward/query/PDE/Greek paths, datasets and evaluation, followed by
retraining. This edition documents that issue; it does not claim to repair it.

The model hardcodes K normalization by 100 and T normalization by 2. Its output
ansatz enforces expiry payoff but does not establish positivity, arbitrage bounds,
or smooth approach to the terminal payoff. Automatic differentiation through the
current feature/query construction also needs numerical verification.

Analytical NumPy pricing supports zero time, zero volatility and zero spot. The
Torch helper is tested for positive regular inputs; zero-volatility derivatives
and expiry Greeks remain outside the tested contract. Existing Greek utilities
have scalar-maturity assumptions. Gamma labels are clipped. No end-to-end training,
GPU benchmark, cloud upload, or checkpoint compatibility study is implied by the
unit tests. European calls are the learned target; utility put pricing does not
make the trained operator a put model.
