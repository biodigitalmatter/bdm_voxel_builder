from bdm_voxel_builder.config_setup import Config

from bdm_voxel_builder.agent_algorithms.algo_8_build_on import Algo8
from bdm_voxel_builder.agent_algorithms.algo_7_queen_box import Algo7QueenBox

from bdm_voxel_builder.visualizers.matplotlib import MPLVisualizer

# SHORT RUN
voxel_size = 60
plot_this = None
# plot_this = ("existing_geo")
# plot_this = ('built_ph_layer', "existing_geo", "agent_space", )

config = Config(
    iterations=400,
    scale=voxel_size,
    algo=Algo7QueenBox(agent_count=20, voxel_size=voxel_size),
    visualizer=MPLVisualizer(
        save_file=True,
        scale=voxel_size,
        save_animation=True,
        color_4d=False,
        selected_layers = plot_this,
        clear=True,
        show_fig=True
    ),
    save_interval=50,
    datalayers_to_visualize=plot_this,
)

# # LONG RUN
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
