module gmmodel
using Distributions
using Plots
using LinearAlgebra


# GMM defintion
mutable struct GMM
    n_components::Integer
    μ::Vector{Number}
    π::Vector{Number}
    Σ::Vector{Number}

    function GMM(n_components::Integer)
        μ = rand(n_components)
        Σ = rand(n_components)
        π = rand(Dirichlet(ones(n_components)))
        new(n_components, μ, π, Σ)
    end
end

mutable struct MvGMM
    n_components:: Integer
    n_variables:: Integer
    μ::Matrix{Float64}
    Σ::Matrix{Float64}
    π::Vector{Number}

    function MvGMM(n_components::Integer, n_variables::Integer)
        μ = rand(n_components, n_variables)
        Σ = rand(n_components, n_variables)
        π = rand(Dirichlet(ones(n_components)))
        new(n_components, n_variables, μ, Σ, π)
    end

    function MvGMM(n_components::Integer, n_variables::Integer, μ::Matrix{Float64}, Σ::Matrix{Float64})
        π = rand(Dirichlet(ones(n_components)))
        new(n_components, n_variables, μ, Σ, π)
    end
end


function fit(x, n_components::Integer, n_iter::Integer=10)
    model = GMM(n_components)
    n_samples = length(x)

    for iter in 1:n_iter
        # E-step
        responsibilites = zeros((n_samples, n_components))
        for s in 1:n_samples
            for c in 1:n_components
                d = Normal(model.μ[c], model.Σ[c])
                responsibilites[s, c] = model.π[c] * pdf(d, x[s])
            end
        end
        responsibilites = responsibilites ./ sum(responsibilites, dims=2)

        c_resp = vec(sum(responsibilites, dims=1))

        # M-Step
        for c in 1:n_components
            #gmm.μ[c] = sum(responsibilites[:, c] .* x) / n_samples
            #gmm.Σ[c] = sum(responsibilites[:, c] .* (x .- gmm.μ[c]) .^ 2) / n_samples
            model.μ[c] = sum(responsibilites[:, c] .* x) / c_resp[c]
            model.Σ[c] = sqrt(sum(responsibilites[:, c] .* (x .- model.μ[c]) .^ 2) / c_resp[c])
            model.π[c] = c_resp[c] ./ n_samples
        end
    end
    model
end

end