#pass
from voxel_builder_library import pheromon_loop, make_solid_box_z, make_solid_box_xxyyzz
from class_agent import Agent
from class_layer import Layer
import numpy as np
from show_voxel_plt import timestamp_now

"""
SETUP GOAL
testing build setups
"""

# MAIN SETTINGS
note = 'setup5reset_build_by_queen_self_collision'
iterations = 500
time__ = timestamp_now
save_json_every_nth = 100
trim_floor = False

### SAVE
save_img = True
save_json = False
save_animation = False
show_animation = False
# img plot type
show_scatter_img_bool = False
show_voxel_img_bool = True
color_4D = False
scale_colors = 1

# select_layers to plot settings
selected_to_plot = [
    'ground'
]

# overal settings
voxel_size = 40
agent_count = 50
wait_to_diffuse = 25

# BUILD SETTINGS
reach_to_build = 0.5
reach_to_erase = 1
stacked_chances = True
reset_after_build = True

# pheromon sensitivity
queen_pheromon_min_to_build = 0.005
queen_pheromon_max_to_build = 0.05
queen_pheromon_build_strength = 1
queen_ph_build_flat_strength = True

# Agent deployment
deployment_zone__a = 5
deployment_zone__b = 35

# MOVE SETTINGS
# pheromon layers
move_ph_random_strength = 0.0001
move_ph_queen_bee_strength = 2
moisture_ph_strength = 0

# direction preference
move_dir_prefer_to_side = 0
move_dir_prefer_to_up = 0
move_dir_prefer_to_down = 0
move_dir_prefer_strength = 0

# general
check_collision = True
keep_in_bounds = True

# PHEROMON SETTINGS
# queen bee:
queens_place = [20,20,2]
queens_place_array = np.zeros([voxel_size, voxel_size, voxel_size])
x,y,z = queens_place
queens_place_array[x][y][z] = 1
queen_bee_pheromon_gravity_ratio = 0

# ENVIRONMENT GEO
ground_level_Z = 1
solid_box = None
# solid_box = [25,26,0,30,ground_level_Z,12]
# solid_box = [10,20,10,20,0,6]
# solid_box = [0,1,0,1,0,1]

def layer_setup(iterations):
    """
    creates the simulation environment setup
    with preset values in the definition
    
    returns: [settings, layers, clai_moisture_layer]
    layers = [agent_space, air_moisture_layer, build_boundary_pheromon, clay_moisture_layer,  ground, queen_bee_pheromon, sky_ph_layer]
    settings = [agent_count, voxel_size]
    """
    ### LAYERS OF THE ENVIRONMENT
    rgb_agents = [34,116,240]
    rgb_ground = [207, 179, 171]
    rgb_queen = [232, 226, 211]
    rgb_queen = [237, 190, 71]



    ground = Layer(voxel_size=voxel_size, name='ground', rgb = [i/255 for i in rgb_ground])
    agent_space = Layer('agent_space', voxel_size = voxel_size, rgb = [i/255 for i in rgb_agents])
    queen_bee_pheromon = Layer('queen_bee_pheromon', voxel_size=voxel_size, rgb = [i/255 for i in rgb_queen], flip_colors = True)

    queen_bee_pheromon.diffusion_ratio = 1/7
    queen_bee_pheromon.decay_ratio = 1/1000
    queen_bee_pheromon.gradient_resolution = 0
    queen_bee_pheromon.gravity_dir = 5
    queen_bee_pheromon.gravity_ratio = queen_bee_pheromon_gravity_ratio

    ### CREATE GROUND
    ground.array[:,:,:ground_level_Z] = 1
    # print(ground.array)
    if solid_box != None:
        wall = make_solid_box_xxyyzz(voxel_size, *solid_box)
        ground.array += wall

    # set ground moisture
    # clay_moisture_layer.array = ground.array.copy()

    # WRAP ENVIRONMENT
    layers = {'agent_space' : agent_space,
              'ground' : ground, 'queen_bee_pheromon' : queen_bee_pheromon}
    settings = {"agent_count" : agent_count, "voxel_size" : voxel_size}
    return settings, layers

def diffuse_environment(layers):
    ground = layers['ground']
    queen_bee_pheromon = layers['queen_bee_pheromon']
    pheromon_loop(queen_bee_pheromon, emmission_array=queens_place_array, blocking_layer=ground, gravity_shift_bool=True)
    pass

def setup_agents(layers):
    agent_space = layers['agent_space']
    ground = layers['ground']
    agents = []
    for i in range(agent_count):
        # create object
        agent = Agent(
            space_layer = agent_space, ground_layer = ground,
            track_layer = None, leave_trace=False, save_move_history=True)
        
        # drop in the middle
        reset_agent(agent)

        agents.append(agent)
    return agents

def reset_agent(agent):
    # centered setup
    a, b = [deployment_zone__a, deployment_zone__b]
    
    x = np.random.randint(a, b)
    y = np.random.randint(a, b)
    z = ground_level_Z

    agent.pose = [x,y,z]

    agent.build_chance = 0
    agent.erase_chance = 0
    agent.move_history = []

def move_agent(agent, layers):
    """moves agents in a calculated direction
    calculate weigthed sum of slices of layers makes the direction_cube
    check and excludes illegal moves by replace values to -1
    move agent
    return True if moved, False if not or in ground
    """
    pose = agent.pose
    # check if agent_in_ground
    
    # # check layer value
    gv = agent.get_layer_value_at_pose(layers['ground'], print_ = False)
    if gv != 0:
        return False

    # move by queen_ph
    layer = layers['queen_bee_pheromon']
    domain = [queen_pheromon_min_to_build, queen_pheromon_max_to_build]
    strength = move_ph_queen_bee_strength
    ph_cube_1 = agent.get_direction_cube_values_for_layer_domain(layer, domain, strength)

    # get random directions cube
    random_cube = np.random.random(26) * move_ph_random_strength

    cube = ph_cube_1 + random_cube
    
    # global direction preference cube
    move_dir_preferences = [
        move_dir_prefer_to_up,
        move_dir_prefer_to_side,
        move_dir_prefer_to_down
    ]
    if move_dir_preferences != None:
        up, side, down = move_dir_preferences
        cube += agent.direction_preference_26_pheromones_v2(up, side, down) * move_dir_prefer_strength
    
    moved = agent.move_on_ground_by_cube(ground=layers['ground'], pheromon_cube=cube, voxel_size=voxel_size, 
                                         fly = False, only_bounds = keep_in_bounds, check_self_collision = check_collision)
    
    # check if in bounds
    if 0 > np.min(agent.pose) or np.max(agent.pose) >= voxel_size :
        # print(agent.pose)
        moved = False

    return moved

def calculate_build_chances_full(agent, layers):
    """PLACEHOLDER NOT REMOVED!
    build_chance, erase_chance = [0.2,0]
    function operating with Agent and Layer class objects
    calculates probability of building and erasing voxels 
    combining several density analyses

    returns build_chance, erase_chance
    """
    ground = layers['ground']
    queen_bee_pheromon = layers['queen_bee_pheromon']

    build_chance = agent.build_chance
    erase_chance = agent.erase_chance

    queen_pheromon_min_to_build = 0.5
    queen_pheromon_build_chance = 1

    # RELATIVE POSITION
    c = agent.get_chance_by_relative_position(
        ground,
        build_below = 2,
        build_aside = 1,
        build_above = 1,
        build_strength = 0.1)
    build_chance += c

    # surrrounding ground_density
    c, e = agent.get_chances_by_density(
            ground,      
            build_if_over = queen_pheromon_min_to_build,
            build_if_below = 27,
            erase_if_over = 27,
            erase_if_below = 0,
            build_strength = queen_pheromon_build_chance)
    build_chance += c
    erase_chance += e

    v = agent.get_pheromone_strength(queen_bee_pheromon, queen_pheromon_min_to_build)
    build_chance += v


    return build_chance, erase_chance

def calculate_build_chances(agent, layers):
    """simple build chance getter, based on 

    returns build_chance, erase_chance
    """
    upper_limit = None
    queen_bee_pheromon = layers['queen_bee_pheromon']

    build_chance = agent.build_chance
    erase_chance = agent.erase_chance

    v = agent.get_pheromone_strength(queen_bee_pheromon, queen_pheromon_min_to_build, queen_pheromon_max_to_build, queen_pheromon_build_strength, queen_ph_build_flat_strength)
    build_chance += v
    erase_chance += 0

    return build_chance, erase_chance

def build_over_limits_old(agent, layers, build_chance, erase_chance, decay_clay = False):
    ground = layers['ground']
    clay_moisture_layer = layers['clay_moisture_layer']
    """agent builds on construction_layer, if chances are higher
    return bool"""
    if stacked_chances:
        # print(erase_chance)
        agent.build_chance += build_chance
        agent.erase_chance += erase_chance
    else:
        agent.build_chance = build_chance
        agent.erase_chance = erase_chance

    # CHECK IF BUILD CONDITIONS are favorable
    built = False
    erased = False
    build_condition = agent.check_build_conditions(ground)
    if agent.build_chance >= reach_to_build and build_condition == True:
        built = agent.build()
        built2 = agent.build_on_layer(clay_moisture_layer)
        if built and reset_after_build:
            # reset_agent = True
            if decay_clay:
                clay_moisture_layer.decay_linear()
    elif agent.erase_chance >= reach_to_erase and build_condition == True:
        erased = agent.erase(ground)
        erased2 = agent.erase(clay_moisture_layer)
    return built, erased

def build_over_limits(agent, layers, build_chance, erase_chance):
    ground = layers['ground']
    """agent builds on construction_layer, if pheromon value in cell hits limit
    chances are either momentary values or stacked by history
    return bool"""
    if stacked_chances:
        # print(erase_chance)
        agent.build_chance += build_chance
        agent.erase_chance += erase_chance
    else:
        agent.build_chance = build_chance
        agent.erase_chance = erase_chance

    # check is there is any solid neighbors
    build_condition = agent.check_build_conditions(ground)

    built = False
    erased = False
    if build_condition:
        # build
        if agent.build_chance >= reach_to_build:
            built = agent.build(ground)
        # erase
        elif agent.erase_chance >= reach_to_erase:
            erased = agent.erase(ground)
    return built, erased

def build_roll_a_dice(agent, layers, build_chance, erase_chance):
    ground = layers['ground']
    """agent builds on construction_layer, if pheromon value in cell hits limit * random value
    chances are either momentary values or stacked by history

    return bool"""
    if stacked_chances:
        # print(erase_chance)
        agent.build_chance += build_chance
        agent.erase_chance += erase_chance
    else:
        agent.build_chance = build_chance
        agent.erase_chance = erase_chance

    # CHECK IF BUILD CONDITIONS are favorable
    built = False
    erased = False
    build_condition = agent.check_build_conditions(ground)
    min_chance_to_build = np.random.random(1) * reach_to_build
    min_chance_to_erase = np.random.random(1) * reach_to_erase
    if agent.build_chance >= min_chance_to_build and build_condition == True:
        built = agent.build(ground)
    elif agent.erase_chance >= min_chance_to_erase and build_condition == True:
        erased = agent.erase(ground)

    return built, erased

def build(agent, layers, build_chance, erase_chance):
    """build - select build style here"""
    # bool_ = build_roll_a_dice(agent, layers, build_chance, erase_chance)
    bool_ = build_over_limits(agent, layers, build_chance, erase_chance)
    # bool_ =  build_over_limits_old(agent, layers, build_chance, erase_chance, decay_clay = True)
    built, erased = bool_
    return built, erased



    
