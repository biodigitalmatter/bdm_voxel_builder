import numpy as np
import numpy.typing as npt

from bdm_voxel_builder.data_layer import DataLayer


def pheromon_loop(
    pheromon_layer: DataLayer,
    emmission_array: npt.NDArray = None,
    n_iterations: int = 1,
    blocking_layer: DataLayer = None,
    gravity_shift_bool: bool = False,
    diffuse_bool: bool = True,
    decay: bool = True,
    decay_linear: bool = False,
):
    """gravity direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up"""
    for _ in range(n_iterations):
        # emmission in
        if isinstance(emmission_array, np.ndarray):
            pheromon_layer.emission_intake(emmission_array, 2, False)

        # diffuse
        if diffuse_bool:
            pheromon_layer.diffuse()

        # gravity
        if gravity_shift_bool:
            pheromon_layer.gravity_shift()

        # decay
        if decay_linear:
            pheromon_layer.decay_linear()
        elif decay:
            pheromon_layer.decay()

        # collision
        if blocking_layer:
            blocking_layer.block_layers([pheromon_layer])

        # apply gradient steps
        if pheromon_layer.gradient_resolution != 0:
            pheromon_layer.grade()
