from bdm_voxel_builder.config_setup import Config

from bdm_voxel_builder.agent_algorithms.algo_7_queen_box import Algo7QueenBox

from bdm_voxel_builder.visualizers.compas_viewer import CompasViewerVisualizer

# SHORT RUN test compasview
voxel_size = 60
plot_this = None
# plot_this = ("existing_geo")
# plot_this = ('built_ph_layer', "existing_geo", "agent_space", )

interval = 50

config = Config(
    iterations=100,
    scale=voxel_size,
    algo=Algo7QueenBox(agent_count=20, voxel_size=voxel_size),
    visualizer=CompasViewerVisualizer(save_file=True),
    save_interval=interval,
    visualize_interval=interval,
    datalayers_to_visualize=plot_this,
)

# # Algo8b testing
# voxel_size = 60
# iterations = 100
# plot_this = None
# plot_this = ('ground')
# plot_this = ("clay_layer", "agent_space")

# config = Config(
#     iterations=iterations,
#     scale=voxel_size,
#     algo=Algo8b(
#         agent_count=100,
#         voxel_size=voxel_size,
#     ),
#     visualizer=MPLVisualizer(
#         save_file=False,
#         scale=voxel_size,
#         save_animation=True,
#         color_4d=False,
#         selected_layers = plot_this,
#         clear=True,
        
#     ),
#     save_interval=iterations-1
# )