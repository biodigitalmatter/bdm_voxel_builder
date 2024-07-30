from compas.data import json_dump
from compas.geometry import Point, Pointcloud
from compas_viewer import Viewer

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers.numpy import convert_array_to_pts
from bdm_voxel_builder.helpers.savepaths import get_savepath
from bdm_voxel_builder.visualizer.base import Visualizer


class CompasViewerVisualizer(Visualizer):
    FILE_SUFFIX = ".json"

    def __init__(self, save_file=False, skip_layers=("layer_name")):
        super().__init__(save_file)

        self.viewer = Viewer()
        self.scene = self.viewer.scene
        self.skip_layers = skip_layers

    def setup_layers(self):
        # set up parent objects for each layer
        for layer in self.data_layers:
            if layer.name not in self.skip_layers and not self.scene.get_node_by_name(
                layer.name
            ):
                pt = Point(0, 0, 0)
                self.scene.add(
                    pt,
                    name=layer.name,
                    pointcolor=layer.color,
                    pointsize=0.1,
                    parent=None,
                )

    def save_file(self, note=None):
        filepath = get_savepath(TEMP_DIR, self.FILE_SUFFIX, note=note)

        json_dump(self.scene, filepath)

    def clear(self):
        self.scene.clear()

    def draw(self, iteration_count=None):
        self.setup_layers()
        for layer in self.data_layers:
            if layer.name in self.skip_layers:
                continue
            parent = self.scene.get_node_by_name(layer.name)

            pts = convert_array_to_pts(layer.array, get_data=False)

            iteration = iteration_count or len(parent.children)
            name = f"{layer.name}_{iteration}"
            self.scene.add(
                Pointcloud(pts), name=name, pointcolor=layer.color, parent=parent
            )

    def show(self):
        self.viewer.show()
