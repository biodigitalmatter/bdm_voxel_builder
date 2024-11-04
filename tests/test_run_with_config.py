from unittest.mock import patch

from bdm_voxel_builder import DATA_DIR
from bdm_voxel_builder.__main__ import _load_config, run_algo
from bdm_voxel_builder.config_setup import Config


@patch(
    "bdm_voxel_builder.visualizer.compas_viewer.CompasViewerVisualizer.__init__",
    return_value=None,
)
def test_run_with_config(mock_viewer):
    config: Config = _load_config(DATA_DIR / "config.py")
    config.iterations = 10
    config.visualizer = None

    run_algo(config)
    mock_viewer.assert_called_once()
