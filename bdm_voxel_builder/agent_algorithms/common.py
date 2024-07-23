def pheromon_loop(
    pheromon_layer,
    emmission_array=None,
    i=1,
    blocking_layer=None,
    gravity_shift_bool=False,
    diffuse_bool=True,
    decay=True,
    decay_linear=False,
):
    """gravity direction: 0:left, 1:right, 2:front, 3:back, 4:down, 5:up"""
    for i in range(i):
        # emmission in
        if not isinstance(emmission_array, bool):
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
