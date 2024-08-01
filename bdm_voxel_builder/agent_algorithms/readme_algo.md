# Algorithm Class Readme

description of the voxel builder algorithms

## generic structure

parameter settings
initialization
    make_layers
    setup_agents
        reset_agents
iterate:
    update_environment
    move_agents
        reset_agents
    calculate_build_chances
    build/erase
        reset_agents

## Algo_8_c_build_on

### Summary

default, basic voxel builder algorithm
agents build on and around an initial 'clay' volume on a 'ground' surface
inputs: solid_ground_volume, clay_volume
output:
    A >> flat growth around 'clay' covering the ground
    B >> tower growth on 'clay'

### Agent behaviour

1. find the built clay
2. climb upwards on it
3. build after a while of climbing
4. reset or not

### Features

- move on solid array
- move direction is controlled with the mix of the pheromon environment and a global direction preference
- move randomness controlled by setting the number of best directions for the random choice
- build on existing volume
- build and erase is controlled by gaining rewards
- move and build both is regulated differently at different levels of environment layer density 

### Observations

resetting the agents after build results in a flat volume, since the agent generally climbs upwards for the same amount of steps
not resetting the agent after build results in towerlike output
more agents > wider tower, less agent > thinner tower. because a. reach the same level simultanously

generally agent count has an effect on the outcome geometry

## Algo_8_d_build_fresh

NEW in 8_d

the clay array values slowly decay
agents aim more towards the freshly built volumes.

main parameter is how fast the clay 'dries'
decay between 1/100 - 1/100000000

agents dont have to necceserily reset, because its enough to reset the build_chance

## Algo_9_a_carve

agents check voxels above, and erase if it denser than 2/3

## varation: Algo_8_b_build_by_density

WIP
old version of 8c, doesnt work as intended
yet
