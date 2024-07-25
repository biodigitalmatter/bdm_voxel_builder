import functools

from compas.geometry import Box
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.helpers.numpy import convert_array_to_pts
from bdm_voxel_builder.helpers.savepaths import get_savepath
from bdm_voxel_builder.simulation_state import SimulationState
from bdm_voxel_builder.visualizers.base import Visualizer


class MPLVisualizer(Visualizer):
    FILE_SUFFIX = ".png"

    def __init__(
        self,
        save_file=False,
        bbox: Box = None,
        save_animation=False,
        scale=20,
        color_4d=False,
    ):
        if not bbox:
            bbox = Box(scale)

        super().__init__(save_file, bbox=bbox)

        self.color_4d = color_4d

        self.should_save_animation = save_animation

        self.setup()

    def setup(self):
        fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"})

        fig.set_size_inches(6, 6)
        fig.set_dpi(300)

        # remove ticks
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])

        ax1.set_axis_off()

        xbounds = self.bbox.xmin, self.bbox.xmax
        ybounds = self.bbox.ymin, self.bbox.ymax
        zbounds = self.bbox.zmin, self.bbox.zmax

        ax1.set_xlim(xbounds)
        ax1.set_ylim(ybounds)
        ax1.set_zlim(zbounds)

        ax1.view_init(elev=30, azim=30, roll=30)

        self.fig = fig
        self.ax1 = ax1

    def save_file(self, note=None):
        filepath = get_savepath(TEMP_DIR, self.FILE_SUFFIX, note=note)

        self.fig.savefig(
            str(filepath),
            bbox_inches="tight",
            dpi=200,
        )

    def setup_animation(
        self,
        func: callable,
        config: Config,
        sim_state: SimulationState,
        iterations: int,
    ):
        self.animation = FuncAnimation(
            self.fig,
            functools.partial(func, config=config, sim_state=sim_state),
            frames=iterations,
            interval=1,
        )

    def save_animation(self, note=None):
        filepath = get_savepath(TEMP_DIR, ".gif", note=note)

        self.animation.save(str(filepath))
        print(f"MPL animation saved to {filepath}")

    def clear(self):
        self.ax1.clear()

    def draw(self):
        for layer in self.data_layers:
            pts = np.array(convert_array_to_pts(layer.array)).transpose()
            self.ax1.scatter(
                pts[0, :],
                pts[1, :],
                pts[2, :],
                marker="s",
                s=1,
                facecolor=layer.color.rgb,
            )

    def show(self):
        self.fig.show()
