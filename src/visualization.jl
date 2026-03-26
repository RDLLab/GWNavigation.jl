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

function POMDPs.simulate(sim::GWNavigationSimulator, pomdp, args...)
    error("To use GWNavigationSimulator, you must have GLMakie loaded. Please run `using GLMakie`.")
end

# Stub function definition
function plot_policy_dic(pomdp::POMDP, policy::Dict{GWState, Symbol})
    error("To use `plot_policy_dic`, you must have GLMakie loaded. Please run `using GLMakie`.")
end

function plot_state_indexs(pomdp::POMDP)
    error("To use `plot_state_indexs`, you must have GLMakie loaded. Please run `using GLMakie`.")
end