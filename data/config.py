from bdm_voxel_builder.config_setup import Config

from bdm_voxel_builder.agent_algorithms.algo_8_build_on import Algo8
# from bdm_voxel_builder.agent_algorithms.algo_7_queen_box import Algo7QueenBox

from bdm_voxel_builder.visualizers.matplotlib import MPLVisualizer

# SMALL RUN
voxel_size = 100
plot_this = None
plot_this = ("agent_space", "existing_geo")
# plot_this = ('built_ph_layer', "existing_geo", "agent_space", )

config = Config(
    iterations=100,
    scale=voxel_size,
    algo=Algo8(
        agent_count=100,
        voxel_size=voxel_size
    ),
    visualizer=MPLVisualizer(
        save_file=True,
        scale=voxel_size,
        save_animation=True,
        color_4d=False,
        selected_layers = plot_this,
        clear=True
    ),
    save_interval=100
)

# # LARGE RUN
# voxel_size = 200
# plot_this = None
# plot_this = ("existing_geo")

# config = Config(
#     iterations=1000,
#     scale=voxel_size,
#     algo=Algo8(
#         agent_count=200,
#         voxel_size=voxel_size
#     ),
#     visualizer=MPLVisualizer(
#         save_file=True,
#         scale=voxel_size,
#         save_animation=False,
#         color_4d=False,
#         selected_layers = plot_this,
#         clear=True
#     ),
#     save_interval=50
# )
