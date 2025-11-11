module GWNavigationSimExt

# This code is only loaded when the user does `using GLMakie`

using POMDPs
using POMDPTools
using GLMakie

# Import your main package to access its types and functions
using GWNavigation

function POMDPs.simulate(sim::GWNavigationSimulator, pomdp::GWNavigationPOMDP, args...)
    figure, ax = draw_the_gridworld(pomdp)
    isinitial = true
    robot_pos = Observable(Point2f(1, 1))
    robot_marker_size = 20

    # Observables for belief visualization
    belief_positions = Observable(Point2f[])
    belief_colours = Observable(Tuple{String,Float64}[])

    display(figure)

    for step in simulate(sim.stepsim, pomdp, args...)
        if isinitial
            isinitial = false
            println("Initial step")
            robot_pos = Observable(Point2f(step.s[1], step.s[2]))
            # Draw the robot
            scatter!(ax, robot_pos, color=:blue, marker=:circle, markersize=robot_marker_size)

            # Draw the initial belief
            positions = Point2f[]
            colours = Tuple{String,Float64}[]
            for belief_s in support(step.b)
                p = pdf(step.b, belief_s)
                if p > 0.0 # Only plot particles with non-zero probability
                    push!(positions, Point2f(belief_s[1], belief_s[2]))
                    push!(colours, ("#FF0000", p * 0.8 + 0.2))  # Scale alpha for visibility
                end
            end
            belief_positions[] = positions
            belief_colours[] = colours
            scatter!(ax, belief_positions, color=belief_colours, marker=:rect, markersize=robot_marker_size)

            sleep(0.5)  # Pause to visualize the first step
            if sim.pause_each_step
                print("Press Enter to continue...")
                readline()
            end
        end
        println("Step t: ", step.t)
        println("Step s: ", step.s)
        println("Step a: ", step.a)
        println("Step sp: ", step.sp)
        println("Step o: ", step.o)
        println("Step r: ", step.r)
        # for s in support(step.bp)
        #     println("  Belief state: ", s, " with probability ", pdf(step.bp, s))
        # end
        robot_pos[] = Point2f(step.sp[1], step.sp[2])
        # Update belief visualization
        positions = Point2f[]
        colours = Tuple{String,Float64}[]
        for belief_s in support(step.bp)
            p = pdf(step.bp, belief_s)
            if p > 0.0 # Only plot particles with non-zero probability
                push!(positions, Point2f(belief_s[1], belief_s[2]))
                push!(colours, ("#FF0000", p * 0.8 + 0.2))  # Scale alpha for visibility
            end
        end
        belief_positions[] = positions
        belief_colours[] = colours

        sleep(0.5)  # Pause to visualize the step
        if sim.pause_each_step
            print("Press Enter to continue...")
            readline()
        end
    end
end


function GWNavigation.plot_astar_policy(pomdp::GWNavigationPOMDP, policy::GWAStarPolicy)
    fig, ax = draw_the_gridworld(pomdp)
    grid_size = pomdp.grid_size

    # Draw policy arrows
    origins = Point2f[]
    directions = Vec2f[]
    for x in 1:grid_size[1]
        for y in 1:grid_size[2]
            state = GWState(x, y)
            if (state in pomdp.obstacle_states)
                continue
            end
            if haskey(policy.actions, state)
                action = policy.actions[state]
                origin = Point2f(x, y)
                direction = Vec2f(0, 0)
                if action == :Up
                    origin += Point2f(0, -0.25)
                    direction = Vec2f(0, 1)
                elseif action == :Down
                    origin += Point2f(0, 0.25)
                    direction = Vec2f(0, -1)
                elseif action == :Left
                    origin += Point2f(0.25, 0)
                    direction = Vec2f(-1, 0)
                elseif action == :Right
                    origin += Point2f(-0.25, 0)
                    direction = Vec2f(1, 0)
                end
                push!(origins, origin)
                push!(directions, direction)
            end
        end
    end
    arrows2d!(ax, origins, directions, tipwidth=5, tiplength=5, shaftlength=0, color="#FF0000")

    display(fig)
    print("Press Enter to exit...")
    readline()
end

function GWNavigation.plot_state_indexs(pomdp::GWNavigationPOMDP)
    fig, ax = draw_the_gridworld(pomdp)

    for s in POMDPs.states(pomdp)
        idx = POMDPs.stateindex(pomdp, s)
        if haskey(pomdp.free_states, s) || haskey(pomdp.danger_states, s)
            color="#000000"
        else
            color="#FFFFFF"
        end
        text!(ax, string(idx), position=(s[1], s[2]), color=color, align = (:center, :center))
    end
    display(fig)
    print("Press Enter to exit...")
    readline()
end

# Visualization function
function draw_the_gridworld(pomdp::GWNavigationPOMDP)
    grid_size = pomdp.grid_size
    fig = Figure(size=(400 * pomdp.scale_factor + 20, 400 * pomdp.scale_factor + 20))
    ax = Axis(fig[1, 1], title="Grid World Navigation", xlabel="X", ylabel="Y", aspect=DataAspect())
    xlims!(ax, 0.5, grid_size[1] + 0.5)
    ylims!(ax, 0.5, grid_size[2] + 0.5)


    # # Draw grid lines
    # for x in 1:grid_size[1]+1
    #     lines!(ax, [x-0.5, x-0.5], [0.5, grid_size[2]+0.5], color=:lightgray)
    # end

    # Draw obstacles
    for obs in pomdp.obstacle_states
        rect = Rect(obs[1] - 0.5, obs[2] - 0.5, 1, 1)
        poly!(ax, rect, color="#000000")   # Black for obstacles
    end

    # Draw landmarks
    for landmark in keys(pomdp.landmark_states)
        rect = Rect(landmark[1] - 0.5, landmark[2] - 0.5, 1, 1)
        poly!(ax, rect, color="#800080")  # Purple for landmarks
        # text!(ax, string(idx), position=(landmark[1], landmark[2]), color=:white, align = (:center, :center))
    end

    # Draw danger zones
    for danger in keys(pomdp.danger_states)
        rect = Rect(danger[1] - 0.5, danger[2] - 0.5, 1, 1)
        poly!(ax, rect, color="#FFFF00")  # Yellow for danger zones
    end

    # Draw goal states
    for goal in keys(pomdp.goal_states)
        rect = Rect(goal[1] - 0.5, goal[2] - 0.5, 1, 1)
        poly!(ax, rect, color="#008000") # Dark green for goal states
    end

    # Draw initial states
    for init in pomdp.initial_states
        rect = Rect(init[1] - 0.5, init[2] - 0.5, 1, 1)
        poly!(ax, rect, color="#808080")  # Gray for initial states
    end

    return fig, ax
end


end # module GWNavigationSimExt