import pytest
import data_generator_large as generator


def test_cli_applies_overrides(monkeypatch):
    seen = []
    monkeypatch.setattr(generator, 'generate_data', seen.append)
    generator.main(['--n_samples','20','--batch_size','4','--output','custom','--seed','7'])
    config = seen[0]
    assert (config.n_samples, config.batch_size_data, config.data_dir, config.seed) == (20,4,'custom',7)


@pytest.mark.parametrize('args', [['--n_samples','0'], ['--n_samples','9'], ['--batch_size','0']])
def test_cli_rejects_empty_splits_or_batches(args):
    with pytest.raises(SystemExit) as error:
        generator.main(args)
    assert error.value.code == 2
