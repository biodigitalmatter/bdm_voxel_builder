import compas

# TODO: This is a fix for a GH drawing issue
# It can probably be removed after 
# https://github.com/compas-dev/compas/pull/1382
if compas.is_grasshopper():
    import compas_ghpython.drawing

    original_func = compas_ghpython.drawing.draw_mesh

    def draw_mesh(
        vertices, faces, color=None, vertex_normals=None, texture_coordinates=None
    ):
        return original_func(
            vertices,
            faces,
            color=None,
            vertex_normals=vertex_normals,
            texture_coordinates=texture_coordinates,
        )

    draw_mesh.__doc__ = original_func.__doc__

    compas_ghpython.drawing.draw_mesh = draw_mesh
    compas_ghpython.draw_mesh = draw_mesh


__all_plugins__ = [
    "compas_voxel_builder.rhino_install",
]
