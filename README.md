# GWNavigation.jl

Implimentation of the Grid World Navigation problem [1] with the [POMDP.jl](https://github.com/JuliaPOMDP/POMDPs.jl) interface.

[1] Kim E, Karunanayake Y, Kurniawati H. Reference-based POMDPs, NeurIPS 23

## Installation

```julia
using Pkg
Pkg.add("https://github.com/RDLLab/GWNavigation.jl.git")
```

## Example

This example uses the implimented visualization simulator, but can be used with any POMDPTools.jl simulator. Visualizer is implemented as an package extantion that depends on the `GLMakie` package.

```julia
using GWNavigation
using ParticleFilters
using POMDPs
using POMDPTools
using GLMakie   # This activates GWNavigationSimExt 

#Default 20x20 GridWorld, Only impliments 20x20, 60x60 Grid Worlds
pomdp = GWNavigationPOMDP(grid_size=20)
# Greedy A* distance based policy implimentation
policy = GWAStarPolicy(pomdp; uniform_weight=0.1)

updater = BootstrapFilter(pomdp, 1000)
simulator = GWNavigationSimulator(max_steps=30)

POMDPs.simulate(simulator, pomdp, policy, updater)
```

## Visualize A* Policy

```julia
using GWNavigation
using POMDPs
using GLMakie   # This activates GWNavigationSimExt, which allows clalling the "GWNavigation.plot_astar_policy" funtion.

pomdp = GWNavigationPOMDP(grid_size=20)
policy = GWAStarPolicy(pomdp; uniform_weight=0.1)

GWNavigation.plot_astar_policy(pomdp, policy)
```