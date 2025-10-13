using GWNavigation
using ParticleFilters
using POMDPs
using POMDPTools
using GLMakie

# Example usage
pomdp = GWNavigationPOMDP(grid_size=60) # Default 20x20 GridWorld
policy = GWAStarPolicy(pomdp; uniform_weight=0.1)

# #Visualize A* policy
# GWNavigation.plot_astar_policy(pomdp, policy)

updater = BootstrapFilter(pomdp, 1000)
simulator = GWNavigationSimulator(max_steps=30)

simulate(simulator, pomdp, policy, updater)

# Use history recorder to get the discounted reward
# simulator = HistoryRecorder(max_steps=10)
# h = simulate(simulator, pomdp, policy, updater)
# println("Discounted Reward: ", discounted_reward(h))