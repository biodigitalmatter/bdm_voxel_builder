import abc
import contextlib

from compas.geometry import Box

from bdm_voxel_builder.grid import Grid


class Visualizer(abc.ABC):
    FILE_SUFFIX: str | None = None

    def __init__(
        self,
        save_file=False,
        should_show: bool = True,
        bbox: Box | None = None,
        skip_grids: tuple[str] | None = None,
    ):
        self.should_save_file = save_file
        self.should_show = should_show
        self.bbox = bbox

        self.grids: list[Grid] = []

        self.skip_grids = skip_grids or ()

    @abc.abstractmethod
    def save_file(self, note=None):
        raise NotImplementedError

    @abc.abstractmethod
    def show(self):
        raise NotImplementedError

    @abc.abstractmethod
    def draw(self, iteration_count=None):
        raise NotImplementedError

    def add_grid(self, grid: Grid):
        if grid not in self.grids:
            self.grids.append(grid)

    def remove_grid(self, grid: Grid):
        with contextlib.suppress(ValueError()):
            self.grids.remove(grid)
