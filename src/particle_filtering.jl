"""
    A functor to postprocess GWNNavigation belief with ParticleFilters.jl

    This filter out zero-probability particles and add observation matching particles to the belief.
"""
struct GWNavigationParticlePostProcessor{P<:POMDP} <: Function
    pomdp::P
end

function (ppp::GWNavigationParticlePostProcessor)(bp, a, o, b, bb, rng)
    n = n_particles(bp)
    new_weight = weight_sum(bp)/n
    postprocess_i = 0

    for i in 1:n
        if weight(bp, i) == 0.0
            while true
                postprocess_i += 1
                if postprocess_i % 1_000_000 == 0
                    @warn "Stuck in GWNavigationParticlePostProcessor while loop for $postprocess_i iterations."                    
                end
                # Sample a particle from previous preprocessed belief
                s = rand(rng, bb)
                sp, o_sampled = @gen(:sp, :o)(ppp.pomdp, s, a, rng)
                # Check if sampled observation matches actual observation o, if so, set the new particle
                if o_sampled == o
                    set_pair!(bp, i, sp => new_weight)
                    break
                end
            end
        end
    end

    return bp
end