import abc
import contextlib

from compas.geometry import Box

from bdm_voxel_builder.grid import Grid


class Visualizer(abc.ABC):
    FILE_SUFFIX: str = None

    def __init__(self, save_file=False, bbox: Box = None):
        self.should_save_file = save_file
        self.bbox = bbox

        self.data_layers: list[Grid] = []

    @abc.abstractmethod
    def save_file(self, note=None):
        raise NotImplementedError

    @abc.abstractmethod
    def show(self):
        raise NotImplementedError

    @abc.abstractmethod
    def draw(self, iteration_count=None):
        raise NotImplementedError

    def add_data_layer(self, data_layer: Grid):
        if data_layer not in self.data_layers:
            self.data_layers.append(data_layer)

    def remove_data_layer(self, data_layer: Grid):
        with contextlib.suppress(ValueError()):
            self.data_layers.remove(data_layer)
