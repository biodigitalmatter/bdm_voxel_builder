from bdm_voxel_builder import DATA_DIR
from bdm_voxel_builder.__main__ import _load_config, run_algo
from bdm_voxel_builder.config_setup import Config


def test_run_with_config():
    config: Config = _load_config(DATA_DIR / "config.py")
    config.iterations = 10
    config.visualizer = None

    run_algo(config)
