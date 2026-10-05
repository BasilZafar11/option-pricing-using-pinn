"""Run from the repository root: python examples/analytical_price.py."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import black_scholes_price

if __name__ == '__main__':
    for kind in ('call', 'put'):
        price = black_scholes_price(100, 100, 1, .2, .05, kind)
        print(f'{kind}: {price:.6f}')
