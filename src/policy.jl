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
function POMDPs.action(p::GWAStarPolicy, b)
    if rand(p.rng) < p.uniform_weight
        return rand(p.rng, POMDPs.actions(p.pomdp))
    else
        s = rand(p.rng, b)
        if s[1] == 0 && s[2] == 0
            # Edge case: if state is (0, 0): Terminal state, choose random action.
            return rand(p.rng, POMDPs.actions(p.pomdp))
        end
        return p.actions[s]
    end
end

# MDP action selection using A*
function POMDPs.action(p::GWAStarPolicy, s::GWState)
    if rand(p.rng) < p.uniform_weight
        return rand(p.rng, POMDPs.actions(p.pomdp))
    else
        if s[1] == 0 && s[2] == 0
            # Edge case: if state is (0, 0): Terminal state, choose random action.
            return rand(p.rng, POMDPs.actions(p.pomdp))
        end
        return p.actions[s]
    end
end

POMDPTools.action_info(p::GWAStarPolicy{<:POMDP}, s) = (POMDPs.action(p, s), nothing)


"""
    GWLocalizeOrAStarPolicy
    Either suggest action to a closest localization point or use A* policy for the GridWorld Navigation POMDP.
    A* policy or closest localization does not consider danger zones or stocahastic transitions.
    If the belief entropy is above a certain threshold, it will choose the action that leads to the closest localization point. Otherwise, it will use the A* policy.
"""
struct GWLocalizeOrAStarPolicy{P<:POMDP} <: Policy
    pomdp::P
    rng::AbstractRNG
    entropy_threshold::Float64
    astar_actions::Dict{GWState, Symbol}
    localize_actions::Dict{GWState, Symbol}
end

function GWLocalizeOrAStarPolicy(pomdp::POMDP; rng::AbstractRNG=Random.GLOBAL_RNG, entropy_threshold::Float64=1.0)
    astar_actions = calculate_a_star_policy(pomdp)
    localize_actions = calculate_a_star_policy(pomdp; goal_states=collect(keys(pomdp.landmark_states)))
    return GWLocalizeOrAStarPolicy(pomdp, rng, entropy_threshold, astar_actions, localize_actions)
end

function POMDPs.action(p::GWLocalizeOrAStarPolicy, b)
    entropy = -sum(pdf(b, x) * log(pdf(b, x)) for x in support(b) if pdf(b, x) > 0)
    # println("Belief entropy: $entropy")
    s = rand(p.rng, b)
    if s[1] == 0 && s[2] == 0
        # Edge case: if state is (0, 0): Terminal state, choose random action.
        return rand(p.rng, POMDPs.actions(p.pomdp))
    end
    if entropy > p.entropy_threshold
        return p.localize_actions[s]
    else
        return p.astar_actions[s]
    end
end


# Utility functions to compute the A* policy
function calculate_a_star_policy(pomdp::GWNavigationPOMDP; goal_states::Vector{GWState}=collect(keys(pomdp.goal_states)))
    distances = bfs_from_goal(pomdp, goal_states)
    policy = Dict{GWState, Symbol}()

    for s in union(keys(pomdp.free_states), keys(pomdp.goal_states), keys(pomdp.landmark_states), keys(pomdp.danger_states))
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

function bfs_from_goal(pomdp::GWNavigationPOMDP, goal_states::Vector{GWState})
    distances = Dict{GWState, Float64}(s => Inf for s in POMDPs.states(pomdp))
    pq = PriorityQueue{GWState, Float64}()

    for goal_state in goal_states
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