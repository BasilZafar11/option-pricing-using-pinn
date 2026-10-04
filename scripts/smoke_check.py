"""Exercise a tiny generated dataset without training or cloud uploads."""
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from config import Config
from data_generator_large import generate_data


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    with tempfile.TemporaryDirectory() as output:
        config = Config()
        config.n_samples = 20
        config.batch_size_data = 5
        config.S_grid_size = 16
        config.t_grid_size = 8
        config.data_dir = output
        config.gcp_bucket_name = None
        generate_data(config)
        for split, size in [('train',16), ('val',2), ('test',2)]:
            with h5py.File(Path(output) / f'{split}.h5') as data:
                assert data['V'].shape == (size,16,8)
                assert np.isfinite(data['V'][:]).all()
        print('Dataset smoke check passed.')


if __name__ == '__main__':
    main()
