module GWNavigation

using DataStructures
using POMDPs
using POMDPTools
using Random
using StaticArrays

export GWNavigationPOMDP, GWState, GWObservation, GWAStarPolicy, GWNavigationSimulator

const GWState = SVector{2, Int}  # (x, y) coordinates   Type alias for state representation

const GWObservation = SVector{2, Int}  # (x, y) coordinates  Type alias for observation representation

struct GWNavigationPOMDP <: POMDP{GWState, Symbol, GWObservation}
    grid_size::Tuple{Int, Int}
    free_states::Dict{GWState, Int} # Mapping states to their indices
    goal_states::Dict{GWState, Int}
    obstacle_states::Set{GWState}
    landmark_states::Dict{GWState, Int}
    danger_states::Dict{GWState, Int}
    initial_states::Set{GWState}
    observation_dict::Dict{GWObservation, Int} # Mapping observations to their indices
    transition_prob::Float64        # Transition probability for orthogonal direction.
    # observation_prob::Float64
    discount_factor::Float64        # 0.99
    danger_state_penalty::Float64   # -100.0
    goal_state_reward::Float64      # 200.0
    step_penalty::Float64           # -1.0
    scale_factor::Int               # 1
end


POMDPs.discount(pomdp::GWNavigationPOMDP) = pomdp.discount_factor


POMDPs.actions(::GWNavigationPOMDP) = [:Up, :Down, :Left, :Right]

function POMDPs.actionindex(pomdp::GWNavigationPOMDP, a::Symbol)
    @assert in(a, actions(pomdp)) "Invalid action"
    return findfirst(x -> x == a, actions(pomdp))
end


# Check if a state is the terminal state
function POMDPs.isterminal(pomdp::GWNavigationPOMDP, s::GWState)
    return haskey(pomdp.goal_states, s) || haskey(pomdp.danger_states, s)
end

function POMDPs.states(pomdp::GWNavigationPOMDP)
    # Combine free_states, goal_states, and landmark_states into a single collection
    return union(keys(pomdp.free_states), keys(pomdp.goal_states), keys(pomdp.landmark_states), keys(pomdp.danger_states))
end

function POMDPs.stateindex(pomdp::GWNavigationPOMDP, s::GWState)
    @assert 1 <= s[1] <= pomdp.grid_size[1] "Invalid state"
    @assert 1 <= s[2] <= pomdp.grid_size[2] "Invalid state"
    # Try to get the state index from any of the dictionaries
    if haskey(pomdp.free_states, s)
        return pomdp.free_states[s]
    elseif haskey(pomdp.goal_states, s)
        return pomdp.goal_states[s]
    elseif haskey(pomdp.landmark_states, s)
        return pomdp.landmark_states[s]
    elseif haskey(pomdp.danger_states, s)
        return pomdp.danger_states[s]
    else
        error("State $(s) not found in any state dictionary.")
    end
end

# Initial state distribution
POMDPs.initialstate(pomdp::GWNavigationPOMDP) = Uniform(pomdp.initial_states)


function POMDPs.observations(pomdp::GWNavigationPOMDP)
    # Make sure to add Null observation when initializing the POMDP.
    return keys(pomdp.observation_dict)
end

function POMDPs.obsindex(pomdp::GWNavigationPOMDP, o::GWObservation)
    # @assert haskey(pomdp.observation_dict, o) "Invalid observation"
    return pomdp.observation_dict[o]
end


# Transition model
function POMDPs.transition(pomdp::GWNavigationPOMDP, s::GWState, a::Symbol)
    if POMDPs.isterminal(pomdp, s)
        return SparseCat([s], [1.0])
    end

    intended_state = move(s, a, pomdp.grid_size)
    orthogonal_state1, orthogonal_state2 = orthogonal_moves(s, a, pomdp.grid_size)

    intended_state_prob = 1 - pomdp.transition_prob
    orthogonal_state1_prob = pomdp.transition_prob * 0.5
    orthogonal_state2_prob = orthogonal_state1_prob
    current_state_prob = 0.0

    # Check if the intended state is valid
    if intended_state in pomdp.obstacle_states || intended_state == s
        # Stay in the same state if hitting an obstacle
        current_state_prob += intended_state_prob
        intended_state_prob = 0.0
    end
    if orthogonal_state1 in pomdp.obstacle_states || orthogonal_state1 == s
        current_state_prob += orthogonal_state1_prob
        orthogonal_state1_prob = 0.0
    end
    if orthogonal_state2 in pomdp.obstacle_states || orthogonal_state2 == s
        current_state_prob += orthogonal_state2_prob
        orthogonal_state2_prob = 0.0
    end

    probability_vec = [intended_state_prob, orthogonal_state1_prob, orthogonal_state2_prob, current_state_prob]
    state_vec = [intended_state, orthogonal_state1, orthogonal_state2, s]

    # Remove zero-probability transitions
    state_vec = state_vec[probability_vec .> 0.0]
    probability_vec = probability_vec[probability_vec .> 0.0]
    # probability_vec ./= sum(probability_vec)  # Normalize

    # Transition probabilities
    return SparseCat(state_vec, probability_vec)
end


# Define the reward function given the state, action, and next state
function POMDPs.reward(pomdp::GWNavigationPOMDP, s::GWState, a::Symbol, s_next::GWState)
    if POMDPs.isterminal(pomdp, s)
        return 0.0
    elseif haskey(pomdp.goal_states, s_next)
        return pomdp.goal_state_reward * pomdp.scale_factor
    elseif haskey(pomdp.danger_states, s_next)
        return pomdp.danger_state_penalty * pomdp.scale_factor
    else
        return pomdp.step_penalty
    end
end

# Define R(s,a)=E[R(s,a,s′)]
function POMDPs.reward(pomdp::GWNavigationPOMDP, s::GWState, a::Symbol)
    r = 0.0
    for (sp, p) in POMDPs.transition(pomdp, s, a)
        r += p * POMDPs.reward(pomdp, s, a, sp)
    end
    return r
end

# Observation model
function POMDPs.observation(pomdp::GWNavigationPOMDP, a::Symbol, sp::GWState)
    if haskey(pomdp.landmark_states, sp)
        neighbors = get_neighbors(sp, pomdp.grid_size)
        num_neighbors = length(neighbors)
        prob = 1.0 / num_neighbors
        return SparseCat(neighbors, fill(prob, num_neighbors))
    else
        # Null observation (no information)
        return SparseCat([GWState(0,0)], [1.0])
    end
end


## Utility functions

function move(state::GWState, action::Symbol, grid_size::Tuple{Int, Int})
    if action == :Up
        return GWState(state[1], min(state[2] + 1, grid_size[2]))
    elseif action == :Down
        return GWState(state[1], max(state[2] - 1, 1))
    elseif action == :Left
        return GWState(max(state[1] - 1, 1), state[2])
    elseif action == :Right
        return GWState(min(state[1] + 1, grid_size[1]), state[2])
    else
        error("Invalid action")
    end
end

function orthogonal_moves(state::GWState, action::Symbol, grid_size::Tuple{Int, Int})
    if action == :Up || action == :Down
        return [move(state, :Left, grid_size), move(state, :Right, grid_size)]
    elseif action == :Left || action == :Right
        return [move(state, :Up, grid_size), move(state, :Down, grid_size)]
    end
end


# Utility function to get valid neighboring states
function get_neighbors(state::GWState, grid_size::Tuple{Int, Int})
    neighbors = SVector{2, Int}[]
    for dx in -1:1
        for dy in -1:1
            nx, ny = state[1] + dx, state[2] + dy
            if 1 <= nx <= grid_size[1] && 1 <= ny <= grid_size[2]
                push!(neighbors, SVector(nx, ny))
            end
        end
    end
    return neighbors
end


include("constructors.jl")
include("policy.jl")
include("visualization.jl")

end # module GWNavigation
