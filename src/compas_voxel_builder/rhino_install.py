from compas.plugins import plugin


@plugin(category="install")
def installable_rhino_packages():
    return ["compas_voxel_builder"]
