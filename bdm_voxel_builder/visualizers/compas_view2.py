import pathlib

import click
import numpy as np
from compas_view2.app import App
from compas.geometry import Pointcloud, Box

from bdm_voxel_builder.helpers.numpy import convert_array_to_pts
from bdm_voxel_builder.visualizers.base import Visualizer

from OpenGL.error import GLError


class CompasViewVisualizer(Visualizer):
    def __init__(self, bbox: Box = None, width=600, height=600):
        super().__init__(bbox=bbox)
        self.viewer = App(width=width, height=height)
        self.viewer.view.camera.rx = -60
        self.viewer.view.camera.rz = 100
        self.viewer.view.camera.ty = -2
        self.viewer.view.camera.distance = 20

    def save_file(self, note=None):
        print("Save using viewer window")

    def show(self):
        return self.viewer.show()

    def clear(self):
        try:
            self.viewer.view.clear()
        except GLError:
            pass

    def draw(self):
        self.clear()
        for data_layer in self.data_layers:
            pts = convert_array_to_pts(data_layer.array, get_data=False)
            self.viewer.add(Pointcloud(pts), pointcolor=data_layer.color)


@click.command()
@click.argument(
    "file", type=click.Path(exists=True, path_type=pathlib.Path), required=False
)
def show_file(file: pathlib.Path = None):
    import compas.data
    from compas.geometry import Pointcloud

    from bdm_voxel_builder import TEMP_DIR
    from bdm_voxel_builder.helpers.numpy import convert_array_to_pts

    if file is None:
        # Find newest .npy file in DATA_DIR
        files = list(TEMP_DIR.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy files found in {TEMP_DIR}")
        file = max(files, key=lambda f: f.stat().st_mtime)

    match file.suffix:
        case ".npy":
            arr = np.load(file)

            pts = convert_array_to_pts(arr, True)
            pointcloud = Pointcloud(pts)

        case ".json":
            geometry = compas.data.json_load(file)

        case _:
            raise ValueError("File type not supported")

    viz = CompasViewVisualizer()
    viz.viewer.add(pointcloud)
    viz.show()


if __name__ == "__main__":
    show_file()
