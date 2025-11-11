using GWNavigation
using Test
using POMDPs
using POMDPTools: Deterministic
using StaticArrays


@testset "GWNavigation.jl" begin
    # Define a 4x4 grid world for testing
    grid_size = (4, 4)
    goal_states_vec = [SVector(4, 4)]
    obstacle_states_vec = [SVector(2, 2), SVector(2, 3)]
    landmark_states_vec = [SVector(1, 1), SVector(4, 1)]
    danger_states_vec = [SVector(3, 4)]
    initial_states_vec = [SVector(1, 2)]

    all_states = Set(SVector(x,y) for x in 1:grid_size[1], y in 1:grid_size[2])
    special_states = union(goal_states_vec, obstacle_states_vec, landmark_states_vec, danger_states_vec)
    free_states_set = setdiff(all_states, special_states)
    
    free_states = Dict(s => i for (i,s) in enumerate(free_states_set))
    
    # Adjust indices for other state types
    offset = length(free_states)
    goal_states = Dict(s => i + offset for (i,s) in enumerate(goal_states_vec))
    offset += length(goal_states)
    landmark_states = Dict(s => i + offset for (i,s) in enumerate(landmark_states_vec))
    offset += length(landmark_states)
    danger_states = Dict(s => i + offset for (i,s) in enumerate(danger_states_vec))

    # Create a comprehensive observation dictionary
    observation_dict = Dict(SVector(0,0) => 1, SVector(-1,-1) => 2) # Null observation
    observations_to_states = Dict{GWObservation, Set{GWState}}(GWNavigation.GWTerminalObservation => Set{GWState}([GWNavigation.GWTerminalState]))
    obs_idx = 3
    for landmark in keys(landmark_states)
        for observable_s in GWNavigation.get_neighbors(landmark, grid_size)
            obs = GWObservation(observable_s)
            if !haskey(observation_dict, obs)
                observation_dict[obs] = obs_idx
                obs_idx += 1
            end
            if haskey(observations_to_states, obs)
                push!(observations_to_states[obs], landmark)
            else
                observations_to_states[obs] = Set{GWState}([landmark])
            end
        end
    end

    pomdp = GWNavigationPOMDP(
        grid_size,
        free_states,
        goal_states,
        Set(obstacle_states_vec),
        landmark_states,
        danger_states,
        Set(initial_states_vec),
        observation_dict,
        observations_to_states,
        0.1,
        0.99,
        -100.0,
        200.0,
        -1.0,
        1
    )

    @testset "POMDPs.discount" begin
        @test discount(pomdp) == 0.99
    end

    @testset "POMDPs.actions" begin
        @test actions(pomdp) == [:Up, :Down, :Left, :Right]
    end

    @testset "POMDPs.actionindex" begin
        @test actionindex(pomdp, :Up) == 1
        @test actionindex(pomdp, :Down) == 2
        @test actionindex(pomdp, :Left) == 3
        @test actionindex(pomdp, :Right) == 4
    end

    @testset "POMDPs.isterminal" begin
        @test isterminal(pomdp, SVector(0, 0)) == true # Terminal state
        @test isterminal(pomdp, SVector(3, 3)) == false
        @test isterminal(pomdp, SVector(4, 4)) == false
        @test isterminal(pomdp, SVector(3, 4)) == false # Danger state is terminal
        @test isterminal(pomdp, GWNavigation.GWTerminalState) == true
    end

    @testset "POMDPs.states" begin
        s = states(pomdp)
        @test length(s) == grid_size[1] * grid_size[2] - length(obstacle_states_vec) + 1 # +1 for terminal state
        @test SVector(1,1) in s
        @test SVector(4,4) in s
        @test SVector(3,4) in s
    end

    @testset "POMDPs.stateindex" begin
        @test 1 <= stateindex(pomdp, SVector(1, 2)) <= length(states(pomdp))
        @test stateindex(pomdp, SVector(4, 4)) == 11    # Goal state should have the 11th index
        @test stateindex(pomdp, GWNavigation.GWTerminalState) == 15    # Terminal state should have the last index
    end

    @testset "POMDPs.initialstate" begin
        @test support(initialstate(pomdp)) == pomdp.initial_states
    end

    @testset "POMDPs.observations" begin
        @test length(observations(pomdp)) == 9 + 1 # 9 landmark observations + null observation
    end

    @testset "POMDPs.obsindex" begin
        @test obsindex(pomdp, GWNavigation.GWNullObservation) == 1
        @test obsindex(pomdp, GWNavigation.GWTerminalObservation) == 2
        @test obsindex(pomdp, SVector(1,1)) == 3
    end

    @testset "move" begin
        @test GWNavigation.move(SVector(2, 2), :Up, grid_size) == SVector(2, 3)
        @test GWNavigation.move(SVector(2, 2), :Down, grid_size) == SVector(2, 1)
        @test GWNavigation.move(SVector(2, 2), :Left, grid_size) == SVector(1, 2)
        @test GWNavigation.move(SVector(2, 2), :Right, grid_size) == SVector(3, 2)
        # Test boundaries
        @test GWNavigation.move(SVector(1, 1), :Left, grid_size) == SVector(1, 1)
        @test GWNavigation.move(SVector(4, 4), :Right, grid_size) == SVector(4, 4)
    end

    @testset "orthogonal_moves" begin
        @test Set(GWNavigation.orthogonal_moves(SVector(2, 2), :Up, grid_size)) == Set([SVector(1, 2), SVector(3, 2)])
        @test Set(GWNavigation.orthogonal_moves(SVector(2, 2), :Left, grid_size)) == Set([SVector(2, 3), SVector(2, 1)])
    end

    @testset "POMDPs.transition" begin
        # Test transition from a regular state
        s = SVector(1, 2)
        a = :Up
        dist = transition(pomdp, s, a)
        @test Set(dist.vals) == Set([SVector(1, 3), SVector(1, 2)])
        @test dist.probs == [0.9, 0.1]

        # Test transition into an obstacle
        s = SVector(1, 2)
        a = :Right
        dist = transition(pomdp, s, a)
        @test Set(dist.vals) == Set([SVector(1, 3), SVector(1, 1), SVector(1, 2)])
        @test dist.probs == [0.05, 0.05, 0.9]

        # Test transition from a goal state (should go to terminal)
        s = SVector(4, 4)
        a = :Up
        dist = transition(pomdp, s, a)
        @test Deterministic(GWNavigation.GWTerminalState) == dist

        # Test transition from a danger state (should go to terminal)
        s = SVector(3, 4)
        a = :Left
        dist = transition(pomdp, s, a)
        @test Deterministic(GWNavigation.GWTerminalState) == dist

        # Test transition from terminal state (should stay in terminal)
        s = SVector(0, 0)
        a = :Down
        dist = transition(pomdp, s, a)
        @test Deterministic(GWNavigation.GWTerminalState) == dist
    end

    @testset "POMDPs.reward" begin
        # Test R(s,a,s')
        @test reward(pomdp, SVector(4, 3), :Up, SVector(4, 4)) == 200.0
        @test reward(pomdp, SVector(3, 3), :UP, SVector(3, 4)) == -100.0
        @test reward(pomdp, SVector(4, 2), :UP, SVector(4, 3)) == -1.0

        # Test R(s,a)
        @test reward(pomdp, SVector(3, 3), :Right) == -5.95
        @test reward(pomdp, SVector(3, 3), :Up) == -90.1
        @test reward(pomdp, SVector(3, 3), :Left) == -5.95
        @test reward(pomdp, SVector(3, 3), :Down) == -1.0
        @test reward(pomdp, SVector(4, 3), :Up) == 179.89999999999998
        @test reward(pomdp, SVector(3, 4), :Right) == 0.0
        @test reward(pomdp, SVector(4, 4), :Right) == 0.0
        @test reward(pomdp, GWNavigation.GWTerminalState, :Up) == 0.0
    end

    @testset "POMDPs.observation" begin
        # Test observation from a landmark state
        sp = SVector(1, 1)
        dist = observation(pomdp, :Up, sp)
        @test dist.vals == [SVector(1, 1), SVector(1, 2), SVector(2, 1), SVector(2, 2)] # 4 neighbors for a corner
        @test dist.probs == [0.25, 0.25, 0.25, 0.25]

        # Test observation from a non-landmark state
        sp = SVector(3, 2)
        dist = observation(pomdp, :Up, sp)
        @test Deterministic(GWNavigation.GWNullObservation) == dist

        # Test observation from goal state
        sp = SVector(4, 4)
        dist = observation(pomdp, :Up, sp)
        @test Deterministic(GWNavigation.GWNullObservation) == dist

        # Test observation from danger state
        sp = SVector(3, 4)
        dist = observation(pomdp, :Up, sp)
        @test Deterministic(GWNavigation.GWNullObservation) == dist

        # Test observation from terminal state
        sp = GWNavigation.GWTerminalState
        dist = observation(pomdp, :Up, sp)
        @test Deterministic(GWNavigation.GWTerminalObservation) == dist
    end

    @testset "get_neighbors" begin
        # Corner
        @test length(GWNavigation.get_neighbors(SVector(1, 1), grid_size)) == 4
        # Edge
        @test length(GWNavigation.get_neighbors(SVector(1, 2), grid_size)) == 6
        # Center
        @test length(GWNavigation.get_neighbors(SVector(3, 3), grid_size)) == 9
    end

end
