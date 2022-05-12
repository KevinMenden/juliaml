using Distributions
using Plots
using LinearAlgebra

# Create Distributions

function create_sample(n::Int=600)
    n_per_dist = Integer(n / 3)
    d1 = Normal(1, 0.2)
    d2 = Normal(3, 0.6)
    d3 = Normal(5, 0.5)
    x1 = rand(d1, n_per_dist)
    x2 = rand(d2, n_per_dist)
    x3 = rand(d3, n_per_dist)
    x = vcat(x1, x2, x3)
end


# GMM defintion
mutable struct GMModel
    n_components::Integer
    μ::Vector{Number}
    π::Vector{Number}
    Σ::Vector{Number}

    function GMModel(n_components::Integer)
        μ = rand(n_components)
        Σ = rand(n_components)
        π = rand(Dirichlet(ones(n_components)))
        new(n_components, μ, π, Σ)
    end
end

function fitgmm(x::Vector{Float64}, n_components::Integer, n_iter::Integer=10)
    gmm = GMModel(n_components)
    n_samples = length(x)

    for iter in 1:n_iter
        # E-step
        responsibilites = zeros((n_samples, n_components))
        for s in 1:n_samples
            for c in 1:n_components
                d = Normal(gmm.μ[c], gmm.Σ[c])
                responsibilites[s, c] = gmm.π[c] * pdf(d, x[s])
            end
        end
        responsibilites = responsibilites ./ sum(responsibilites, dims=2)

        c_resp = vec(sum(responsibilites, dims=1))

        # M-Step
        for c in 1:n_components
            #gmm.μ[c] = sum(responsibilites[:, c] .* x) / n_samples
            #gmm.Σ[c] = sum(responsibilites[:, c] .* (x .- gmm.μ[c]) .^ 2) / n_samples
            gmm.μ[c] = sum(responsibilites[:, c] .* x) / c_resp[c]
            gmm.Σ[c] = sqrt(sum(responsibilites[:, c] .* (x .- gmm.μ[c]) .^ 2) / c_resp[c])
            gmm.π[c] = c_resp[c] ./ n_samples
        end
    end
    gmm
end

x = create_sample(600)
gmm = fitgmm(x, 3, 50)

