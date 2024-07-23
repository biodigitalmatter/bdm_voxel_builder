from bdm_voxel_builder.simulation_state import SimulationState


class Visualizer:
    SUFFIX: str = None

    def __init__(self, save_file=False, filename=None):
        self.should_save_file = save_file
        self.filename = filename

    def save_file(self, note=None):
        raise NotImplementedError

    def update(self, state: SimulationState):
        raise NotImplementedError

    def show():
        raise NotImplementedError
