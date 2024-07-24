import functools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from bdm_voxel_builder import TEMP_DIR
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.helpers.savepaths import get_savepath
from bdm_voxel_builder.simulation_state import SimulationState
from bdm_voxel_builder.visualizers.base import Visualizer
from bdm_voxel_builder.helpers.numpy import convert_array_to_points


class MPLVisualizer(Visualizer):
    FILE_SUFFIX = ".png"

    def __init__(
        self,
        save_file=False,
        save_animation=False,
        scale=1,
        color_4d=False,
        trim_below=0,
    ):
        super().__init__(save_file)

        self.fig = plt.figure(figsize=[4, 4], dpi=200)
        self.axes = plt.axes(
            xlim=(0, scale), ylim=(0, scale), zlim=(0, scale), projection="3d"
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_zticks([])

        self.color_4d = color_4d
        self.trim_below = trim_below

        self.should_save_animation = save_animation

    def save_file(self, note=None):
        filepath = get_savepath(TEMP_DIR, self.FILE_SUFFIX, note=note)

        plt.savefig(
            str(filepath),
            bbox_inches="tight",
            dpi=200,
        )

    def setup_animation(
        self, func: callable, config: Config, sim_state: SimulationState, iterations: int
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

    def update(self, state: SimulationState):
        for layer in state.data_layers.values():
            if self.color_4d:
                facecolor = layer.color_array[:, :, :, self.trim_below :]
                facecolor = np.clip(facecolor, 0, 1)
            else:
                facecolor = layer.color.rgb
            # scatter plot
            a1 = layer.array.copy()
            pt_array = convert_array_to_points(
                a1[:, :, self.trim_below :], list_output=False
            )
            p = pt_array.transpose()
            self.axes.scatter(
                p[0, :], p[1, :], p[2, :], marker="s", s=1, facecolor=facecolor
            )

    def show(self):
        self.fig.show()
