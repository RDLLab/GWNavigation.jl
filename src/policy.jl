# A* policy for the GridWorld Navigation POMDP

"""
    GWAStarPolicy
    A* policy for the GridWorld Navigation POMDP.
    Chooses actions based on a precomputed A* policy with an option for uniform random actions.
    A* policy does not consider danger zones or stocahastic transitions.
"""
struct GWAStarPolicy{P<:POMDP} <: Policy
    pomdp::P
    rng::AbstractRNG
    uniform_weight::Float64
    actions::Dict{GWState, Symbol}
end

function GWAStarPolicy(pomdp::POMDP; rng::AbstractRNG=Random.GLOBAL_RNG, uniform_weight::Float64=0.0)
    actions = calculate_a_star_policy(pomdp)
    return GWAStarPolicy(pomdp, rng, uniform_weight, actions)
end

# POMDP action selection using A*
function POMDPs.action(p::GWAStarPolicy, b::Union{ParticleCollection{GWState}, WeightedParticleBelief{GWState}})
    if rand(p.rng) < p.uniform_weight
        return rand(p.rng, POMDPs.actions(p.pomdp))
    else
        s = rand(p.rng, b)
        return p.actions[s]
    end
end

# MDP action selection using A*
function POMDPs.action(p::GWAStarPolicy, s::GWState)
    if rand(p.rng) < p.uniform_weight
        return rand(p.rng, POMDPs.actions(p.pomdp))
    else
        return p.actions[s]
    end
end

POMDPTools.action_info(p::GWAStarPolicy{<:POMDP}, s) = (POMDPs.action(p, s), nothing)


# Utility functions to compute the A* policy
function calculate_a_star_policy(pomdp::GWNavigationPOMDP)
    distances = bfs_from_goal(pomdp)
    policy = Dict{GWState, Symbol}()

    for s in POMDPs.states(pomdp)
        if haskey(pomdp.goal_states, s)
            continue
        end

        best_action = :Up
        min_dist = Inf

        for a in POMDPs.actions(pomdp)
            sp = move(s, a, pomdp.grid_size)
            if !(sp in pomdp.obstacle_states) && distances[sp] < min_dist
                min_dist = distances[sp]
                best_action = a
            end
        end
        policy[s] = best_action
    end

    return policy
end

function bfs_from_goal(pomdp::GWNavigationPOMDP)
    distances = Dict{GWState, Float64}(s => Inf for s in POMDPs.states(pomdp))
    pq = PriorityQueue{GWState, Float64}()

    for goal_state in keys(pomdp.goal_states)
        distances[goal_state] = 0
        enqueue!(pq, goal_state, 0)
    end

    while !isempty(pq)
        s = dequeue!(pq)
        
        # Corrected approach: For state `s`, find its predecessors.
        # A state `p` is a predecessor of `s` if `move(p,a) == s` for some action `a`.
        # This is equivalent to `s` being a successor of `p`.
        # Let's check the 4 cells around `s`: up, down, left, right.
        
        potential_predecessors = [
            (s + SVector(0, 1)), # came from Up
            (s - SVector(0, 1)), # came from Down
            (s - SVector(1, 0)), # came from Left
            (s + SVector(1, 0))  # came from Right
        ]
        
        for p in potential_predecessors
            if 1 <= p[1] <= pomdp.grid_size[1] && 1 <= p[2] <= pomdp.grid_size[2] && !(p in pomdp.obstacle_states)
                # if p is a valid state
                if distances[s] + 1 < distances[p]
                    distances[p] = distances[s] + 1
                    pq[p] = distances[p]
                end
            end
        end
    end

    return distances
end