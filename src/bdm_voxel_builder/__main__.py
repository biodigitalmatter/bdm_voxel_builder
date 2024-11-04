import importlib.util
import pathlib

import click

from bdm_voxel_builder import DATA_DIR
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.environment import Environment


def simulate(config: Config, sim_state: Environment):
    algo = config.algo
    visualizer = config.visualizer

    # 1. diffuse environment's grid
    algo.update_environment(sim_state)

    # 2. MOVE and BUILD
    for agent in sim_state.agents:
        # execute move and build actions
        algo.agent_action(agent, sim_state)

    # 3. make frame for animation
    if visualizer and (
        sim_state.iteration_count % config.visualize_interval == 0
        or sim_state.iteration_count == config.iterations - 1
    ):
        visualizer.draw(iteration_count=sim_state.iteration_count)

    # 4. DUMP JSON

    if (
        sim_state.iteration_count % config.save_interval == 0
        or sim_state.iteration_count == config.iterations - 1
    ):
        # a1 = sim_state.grids[algo.grid_to_dump].array.copy()
        # # a1[:, :, : algo.ground_level_Z] = 0

        # save_ndarray(a1, note=note)

        # pointcloud, values = pointcloud_from_grid_array(a1, return_values=True)
        # save_pointcloud(pointcloud, values=values, note=note)
        # save vdb only
        for grid_name in algo.grids_to_dump:
            sim_state.grids[grid_name].save_vdb()
    print(sim_state.iteration_count)
    sim_state.iteration_count += 1


def _load_config(config_file: pathlib.Path) -> Config:
    module_spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(config_module)

    return config_module.config


@click.command()
@click.argument(
    "config-file", type=click.Path(exists=True), default=DATA_DIR / "config.py"
)
def main(config_file):
    config: Config = _load_config(config_file)

    algo = config.algo
    visualizer = config.visualizer
    iterations = config.iterations

    sim_state = Environment(config)

    algo.initialization(sim_state)

    note = f"{algo.name}_a{algo.agent_count}_i{config.iterations}"

    if visualizer:
        for grid in sim_state.grids.values():
            if config.grids_to_visualize:
                for grid_name in config.grids_to_visualize:
                    if grid.name == grid_name:
                        visualizer.add_grid(grid)
            else:
                visualizer.add_grid(grid)

    for _ in range(iterations):
        simulate(config=config, sim_state=sim_state)

    if visualizer:
        if visualizer.should_save_file:
            visualizer.save_file(note=note)

        if visualizer.should_show:
            visualizer.show()


if __name__ == "__main__":
    main()
