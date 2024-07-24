# algorithm readme

## Algorithm structure overview:

1. settings
2. initialization
    create_environment
        (load_scanned_volume)
    setup_agents
        reset_agents

3. iterate:
    1. move_agents
        reset_agents
    2. calculate_build_probability
    3. build/erase
        reset_agents
    4. update_environment
    5. export built array

### notes

## Algo_8

basic build_on existing algorithm

agent is attracted toward existing + newly built geomoetry by 'built_ph_layer'
build_chance is rewarded if within the given ph limits
if enough chances gained, agent builds

### next steps

Objective: Find features by topology and start build on that

default algorithm

grow on attractive features of existing/scanned volumes

### ...

- an initially defined volume attracts the agents by emmitting pheromons
- agents move there
- when close enough, they just randomly move around
- while conducting topology analysis
- once on good topology feature, they build / or their build probability increases 
- do they reset after?
- should newly built add to pheromon emmission stronger?

... should the a build probability be another DiffusiveLayer?, created by agents during theyre move?
effecting their movement or not?

topology features:

1. inner edge +++
2. inner edge undercut ++++++
2. hole +++
3. outer edge
4. below overhang
5. on thin slab?
6. spikes
7. bridge....
8. wall