import numpy as np
import numpy.typing as npt

from bdm_voxel_builder.grid import DiffusiveGrid


def diffuse_diffusive_grid(
    grid: DiffusiveGrid,
    emmission_array: npt.NDArray | list = None,
    blocking_grids: DiffusiveGrid | list[DiffusiveGrid] = None,
    gravity_shift_bool: bool = False,
    diffuse_bool: bool = True,
    decay: bool = True,
    decay_linear: bool = False,
    grade=True,
    n_iterations: int = 1,
):
    """
    DIFFUSE AND DECAY GRIDS
    optionally multiple iterations
    diffusion steps:

    loads from emissive_grids
    diffuse
    apply gravity shift
    decay_linear
    decay_propotional
    get_blocked_by (one or more grids)
    apply gradient resolution

    gravity direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up"""
    for _ in range(n_iterations):
        if isinstance(emmission_array, np.ndarray):
            grid.emission_intake(emmission_array, 1, False)
        elif isinstance(emmission_array, list):
            for i in range(emmission_array):
                grid.emission_intake(emmission_array[i], 1, False)

        # diffuse
        if diffuse_bool:
            grid.diffuse()

        # gravity
        if gravity_shift_bool:
            grid.gravity_shift()

        # decay
        if decay_linear:
            grid.decay_linear()
        elif decay:
            grid.decay()

        # collision

        if blocking_grids:
            if isinstance(blocking_grids, list):
                for blocking_grid in blocking_grids:
                    blocking_grid.block_grids([grid])
            else:
                blocking_grids.block_grids([grid])

        # apply gradient steps
        if grid.gradient_resolution != 0 and grade:
            grid.grade()
