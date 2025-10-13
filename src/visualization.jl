# functionality to visualize the Grid World Navigation POMDP
# Only defined the types
# functionality is implimented in ext/GWNavigationSimExt.jl

struct GWNavigationSimulator <: Simulator
    stepsim::StepSimulator
    pause_each_step::Bool
end

function GWNavigationSimulator(; max_steps::Int=nothing, rng::AbstractRNG=Random.default_rng(), pause_each_step::Bool=true)
    spec = tuple(:s, :a, :sp, :o, :r, :info, :t, :action_info, :b, :bp, :update_info)
    return GWNavigationSimulator(StepSimulator(rng, max_steps, spec), pause_each_step)
end

# Stub function definition
function plot_astar_policy(pomdp, policy)
    error("To use `plot_astar_policy`, you must have GLMakie loaded. Please run `using GLMakie`.")
end