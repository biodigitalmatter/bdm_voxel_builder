import importlib.util
import pathlib

import click

from bdm_voxel_builder import DATA_DIR
from bdm_voxel_builder.config_setup import Config
from bdm_voxel_builder.helpers.numpy import save_ndarray
from bdm_voxel_builder.simulation_state import SimulationState
from bdm_voxel_builder.visualizers.matplotlib import MPLVisualizer

from bdm_voxel_builder.helpers.compas import pointcloud_from_ndarray, save_pointcloud


def simulate(frame, algo, sim_state, config):
    # 1. diffuse environment's layers
    algo.diffuse_environment(sim_state.layers)

    # 2. MOVE and BUILD
    for agent in sim_state.agents:
        # MOVE
        moved = algo.move_agent(agent, sim_state.layers)
        # print(agent.pose)
        # print(moved)
        if not moved:
            algo.reset_agent(agent)
        # BUILD DEMO
        # if moved:
        #     if np.random.random(1) >= 0:
        #         x,y,z = agent.pose
        #         ground.array[x,y,z] = 1
        # BUILD
        if moved:
            build_chance, erase_chance = algo.calculate_build_chances(
                agent, sim_state.layers
            )
            built, erased = algo.build(
                agent, sim_state.layers, build_chance, erase_chance
            )
            # if built:
            #     print('built:', agent.pose)
            if built and algo.reset_after_build:
                algo.reset_agent(agent)
    # 2.b clay dries

    # 3. make frame for animation
    if config.visualizer:
        config.visualizer.update()
    sim_state.counter += 1

    # 4. DUMP JSON
    if config.save_voxel_array:
        note = f"{algo.name}_a{algo.agent_count}_i{algo.iterations}"

        if sim_state.counter % algo.save_json_every_nth == 0:
            a1 = sim_state.data_layers["ground"].array.copy()
            a1[:, :, : algo.ground_level_Z] = 0

            save_ndarray(a1, note=note)

            pointcloud, values = pointcloud_from_ndarray(a1, return_values=True)
            save_pointcloud(pointcloud, values=values, note=note)

    print(sim_state.counter)


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
    config = _load_config(configfile)

    algo = config.algo
    visualizer = config.visualizer
    iterations = config.iterations

    sim_state = SimulationState(config)

    note = f"{algo.name}_a{algo.agent_count}_i{algo.iterations}"

    if isinstance(visualizer, MPLVisualizer) and visualizer.create_animation:
        visualizer.setup_animation(simulate, sim_state=sim_state, iterations=iterations)

        visualizer.update(sim_state)

        if visualizer.save_animation_flag:
            visualizer.save_animation(note=note)

            visualizer.save_file(note=note)

        visualizer.show()

    else:
        for _ in range(algo.iterations):
            simulate(None)

        if visualizer:
            visualizer.update()

            if visualizer.should_save_file:
                visualizer.save_file()

            visualizer.show()


if __name__ == "__main__":
    main()
