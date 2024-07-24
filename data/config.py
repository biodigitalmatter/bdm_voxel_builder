from bdm_voxel_builder.config_setup import Config

from bdm_voxel_builder.agent_algorithms.algo_8_build_on import Algo8
# from bdm_voxel_builder.agent_algorithms.algo_7_queen_box import Algo7QueenBox

from bdm_voxel_builder.visualizers.matplotlib import MPLVisualizer

iterations = 100
voxel_size = 60
agent_count = 200
plot_this = None
plot_this = ("agent_space", "existing_geo")
# plot_this = ('built_ph_layer', "existing_geo", "agent_space", )

config = Config(
    iterations=iterations,
    scale=voxel_size,
    algo=Algo8(agent_count=agent_count, voxel_size=voxel_size),
    visualizer=MPLVisualizer(
        save_file=False,
        scale=voxel_size,
        save_animation=True,
        color_4d=False,
        selected_layers = plot_this,
        clear=True
    ),
    save_interval=iterations+2
)
