include("models/GMM.jl")
import .gmmodel

using Distributions
using Plots
using LinearAlgebra
using DataFrames
using StatsPlots

# Test univariate case
# function create_sample(n::Int=600)
#     n_per_dist = Integer(n / 3)
#     d1 = Normal(1, 0.2)
#     d2 = Normal(3, 0.6)
#     d3 = Normal(5, 0.5)
#     x1 = rand(d1, n_per_dist)
#     x2 = rand(d2, n_per_dist)
#     x3 = rand(d3, n_per_dist)
#     x = vcat(x1, x2, x3)
# end

# x = create_sample(600)
# model = gmmodel.fit(x, 3, 50)


#  Prepare test data for the multivariate case
d1 = MvNormal([1, 1], [0.2, 0.2])
d2 = MvNormal([2, 1], [0.1, 0.9])
x1 = rand(d1, 100)
x2 = rand(d2, 100)
x = hcat(x1, x2)
component = vcat(ones(100), ones(100) .*2)
df = DataFrame(transpose(x), :auto)
df[!, "component"] = component

@df df scatter(:x1, :x2, group= :component)


mvgmm = gmmodel.MvGMM(3, 2)

function sample_mvgmm(model::gmmodel.MvGMM, n_samples::Integer = 100)
    sigma = model.Σ
    mu = model.μ

    s = sigma[1,:]
    m = mu[1,:]
    d = MvNormal(m, s)
    x = rand(d, n_samples)
    component = ones(n_samples)

    for i in 2:size(sigma)[1]
        s = sigma[i,:]
        m = mu[i,:]
        d = MvNormal(m, s)
        x = cat(x, rand(d, n_samples), dims=2)
        component = vcat(component, ones(n_samples) * i)
    end
    df = DataFrame(transpose(x), :auto)
    df[!, "component"] = component
    df
end

df = sample_mvgmm(mvgmm)
@df df scatter(:x1, :x2, group= :component)
