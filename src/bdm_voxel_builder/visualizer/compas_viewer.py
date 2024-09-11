from compas.data import json_dump
from compas.geometry import Point
from compas_viewer import Viewer

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.helpers import get_savepath
from bdm_voxel_builder.visualizer.base import Visualizer


class CompasViewerVisualizer(Visualizer):
    FILE_SUFFIX = ".json"

    def __init__(self, save_file=False, skip_grids: tuple[str] | None = None):
        super().__init__(save_file, skip_grids=skip_grids)

        self.viewer = Viewer()
        self.scene = self.viewer.scene

    def setup_grids(self):
        # set up parent objects for each grids
        for grid in self.grids:
            if grid.name not in self.skip_grids and not self.scene.get_node_by_name(
                grid.name
            ):
                pt = Point(0, 0, 0)
                self.scene.add(
                    pt,
                    name=grid.name,
                    pointcolor=grid.color,
                    pointsize=0.1,
                    parent=None,
                )

    def save_file(self, note=None):
        filepath = get_savepath(TEMP_DIR, self.FILE_SUFFIX, note=note)

        json_dump(self.scene, filepath)

    def clear(self):
        self.scene.clear()

    def draw(self, iteration_count=None):
        self.setup_grids()
        for grid in self.grids:
            if grid.name in self.skip_grids:
                continue
            parent = self.scene.get_node_by_name(grid.name)

            pc = grid.get_world_pointcloud()

            iteration = iteration_count or len(parent.children)
            name = f"{grid.name}_{iteration}"
            self.scene.add(pc, name=name, pointcolor=grid.color, parent=parent)

    def show(self):
        self.viewer.show()
