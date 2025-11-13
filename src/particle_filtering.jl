"""
    A functor to postprocess GWNNavigation belief with ParticleFilters.jl

    This filter out zero-probability particles and add observation matching particles to the belief.
"""
struct GWNavigationParticlePostProcessor{P<:POMDP} <: Function
    pomdp::P
end

function (ppp::GWNavigationParticlePostProcessor)(bp, a, o, b, bb, rng)
    n = n_particles(bp)
    weight_sum = weight_sum(bp)
    if weight_sum == 0.0
        error("All particles have zero weight after observation update. Cannot postprocess belief. a: $a, o: $o")
    else
        # Remove zero-weight particles
        new_weight = weight_sum/n
        postprocess_i = 0

        for i in 1:n
            if weight(bp, i) == 0.0
                if o == GWNullObservation
                    while true
                        postprocess_i += 1
                        if postprocess_i % 1_000_000 == 0
                            @warn "Stuck in GWNavigationParticlePostProcessor while loop for $postprocess_i iterations."                    
                        end
                        # Sample a particle from previous preprocessed belief
                        s = rand(rng, bb)
                        sp, o_sampled = @gen(:sp, :o)(ppp.pomdp, s, a, rng)
                        # Check if sampled observation matches actual observation o, if so, set the new particle
                        if o_sampled == GWNullObservation
                            set_pair!(bp, i, sp => new_weight)
                            break
                        end
                    end
                else
                    # Sample a state from possible states for the observation
                    sp = rand(rng, ppp.pomdp.observations_to_states[o])
                    set_pair!(bp, i, sp => new_weight)
                end
            end
        end
    end

    return bp
end