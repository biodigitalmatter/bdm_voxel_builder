import importlib.util
import pathlib

import click

from bdm_voxel_builder import DATA_DIR
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.helpers import (
    pointcloud_from_ndarray,
    save_ndarray,
    save_pointcloud,
)
from bdm_voxel_builder.visualizer.matplotlib import MPLVisualizer


def simulate(frame, config: Config = None, sim_state: Environment = None):
    algo = config.algo
    visualizer = config.visualizer

    # 1. diffuse environment's grid
    algo.update_environment(sim_state)

    # 2. MOVE and BUILD
    for agent in sim_state.agents:
        # execute move and build actions
        algo.agent_action(agent, sim_state)

    # 3. make frame for animation
    if (
        sim_state.iteration_count % config.visualize_interval == 0
        or sim_state.iteration_count == config.iterations - 1
    ):
        visualizer.draw(iteration_count=sim_state.iteration_count)

    # 4. DUMP JSON
    note = f"{algo.name}_a{algo.agent_count}_i{config.iterations}"

    if (
        sim_state.iteration_count % config.save_interval == 0
        or sim_state.iteration_count == config.iterations - 1
    ):
        a1 = sim_state.grids[algo.grid_to_dump].array.copy()
        a1[:, :, : algo.ground_level_Z] = 0

        save_ndarray(a1, note=note)

        pointcloud, values = pointcloud_from_ndarray(a1, return_values=True)
        save_pointcloud(pointcloud, values=values, note=note)

        sim_state.grids[algo.grid_to_dump].save_vdb()

    print(sim_state.iteration_count)
    sim_state.iteration_count += 1


def _load_config(configfile: pathlib.Path) -> Config:
    module_spec = importlib.util.spec_from_file_location("config", configfile)
    config_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(config_module)

    return config_module.config


@click.command()
@click.argument(
    "configfile", type=click.Path(exists=True), default=DATA_DIR / "config.py"
)
def main(configfile):
    config: Config = _load_config(configfile)

    algo = config.algo
    visualizer = config.visualizer
    iterations = config.iterations

    sim_state = Environment(config)

    note = f"{algo.name}_a{algo.agent_count}_i{config.iterations}"

    for grid in sim_state.grids.values():
        if config.grids_to_visualize:
            for grid_name in config.grids_to_visualize:
                if grid.name == grid_name:
                    visualizer.add_grid(grid)
        else:
            visualizer.add_grid(grid)

    if isinstance(visualizer, MPLVisualizer) and visualizer.should_save_animation:
        visualizer.setup_animation(
            simulate, config=config, sim_state=sim_state, iterations=iterations
        )

        visualizer.save_animation(note=note)

    else:
        for _ in range(config.iterations):
            simulate(None, config=config, sim_state=sim_state)

        if visualizer:
            if visualizer.should_save_file:
                visualizer.save_file(note=note)

            visualizer.show()


if __name__ == "__main__":
    main()
