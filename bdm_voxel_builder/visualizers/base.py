from bdm_voxel_builder.simulation_state import SimulationState


class Visualizer:
    FILE_SUFFIX: str = None

    def __init__(self, save_file=False):
        self.should_save_file = save_file

    def save_file(self, note=None):
        raise NotImplementedError

    def update(self, state: SimulationState):
        raise NotImplementedError

    def show():
        raise NotImplementedError
