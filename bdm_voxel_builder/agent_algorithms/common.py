import numpy as np
import numpy.typing as npt

from bdm_voxel_builder.data_layer.diffusive_layer import DiffusiveLayer


def diffuse_diffusive_layer(
    diffusive_layer: DiffusiveLayer,
    emmission_array: npt.NDArray | list = None,
    blocking_layer: DiffusiveLayer | list = None,
    gravity_shift_bool: bool = False,
    diffuse_bool: bool = True,
    decay: bool = True,
    decay_linear: bool = False,
    grade = True,
    n_iterations: int = 1,
):
    """
    DIFFUSE AND DECAY LAYER
    optionally multiple iterations
    diffusion steps:

    loads from emissive_layers
    diffuse
    apply gravity shift
    decay_linear
    decay_propotional
    get_blocked_by (one or more layers)
    apply gradient resolution

    gravity direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up"""
    for _ in range(n_iterations):

        if isinstance(emmission_array, np.ndarray):
            diffusive_layer.emission_intake(emmission_array, 1, False)
        elif isinstance(emmission_array, list):
            for i in range(emmission_array):
                diffusive_layer.emission_intake(emmission_array[i], 1, False)

        # diffuse
        if diffuse_bool:
            diffusive_layer.diffuse()

        # gravity
        if gravity_shift_bool:
            diffusive_layer.gravity_shift()

        # decay
        if decay_linear:
            diffusive_layer.decay_linear()
        elif decay:
            diffusive_layer.decay()

        # collision
        
        if blocking_layer:
            if isinstance(blocking_layer, list):
                for i in range(len(blocking_layer)):
                    blocking_layer.block_layers(diffusive_layer[i])
            else:
                blocking_layer.block_layers([diffusive_layer])

        # apply gradient steps
        if diffusive_layer.gradient_resolution != 0 and grade:
            diffusive_layer.grade()
