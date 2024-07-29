import importlib.util
import pathlib

import click

from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder import DATA_DIR
from bdm_voxel_builder.helpers.numpy import save_ndarray
from bdm_voxel_builder.environment import Environment
from bdm_voxel_builder.visualizer.matplotlib import MPLVisualizer

from bdm_voxel_builder.helpers.compas import pointcloud_from_ndarray, save_pointcloud

def simulate(frame, config: Config = None, sim_state: Environment = None):
    algo = config.algo
    visualizer = config.visualizer

    # 1. diffuse environment's layers
    algo.update_environment(sim_state)

    # 2. MOVE and BUILD
    for agent in sim_state.agents:
        # MOVE
        moved = algo.move_agent(agent, sim_state)
        if not moved:
            algo.reset_agent(agent)
        # BUILD
        if moved:
            algo.calculate_build_chances(agent, sim_state)
            built, erased = algo.build_by_chance(agent, sim_state)
            if built and algo.reset_after_build:
                algo.reset_agent(agent)
            if erased and algo.reset_after_erased:
                algo.reset_agent(agent)

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
        layer_to_dump = algo.layer_to_dump
        a1 = sim_state.data_layers[layer_to_dump].array.copy()
        a1[:, :, : algo.ground_level_Z] = 0

        save_ndarray(a1, note=note)

        pointcloud, values = pointcloud_from_ndarray(a1, return_values=True)
        save_pointcloud(pointcloud, values=values, note=note)

        sim_state.data_layers[layer_to_dump].save_vdb()

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

    for layer in sim_state.data_layers.values():
        if config.datalayers_to_visualize:
            for layer_name in config.datalayers_to_visualize:
                if layer.name == layer_name:
                    visualizer.add_data_layer(layer)
        else:
            visualizer.add_data_layer(layer)

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
