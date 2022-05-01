
"""PCA implementation."""

using LinearAlgebra
using Random
using Statistics
using MLJ
using Plots
import DataFrames


function PCA(X::Matrix{<:Number}, n_components::Integer=2)
    X_norm = X .- mean(X)
    χ = cov(X_norm, dims=1)
    evals = eigvals(χ)
    evecs = eigvecs(χ)

    sort_index = sortperm(evals, rev=true)
    evecs = evecs[:, sort_index]
    evecs_sub = evecs[:, 1:n_components]
    transpose(transpose(evecs_sub) * transpose(X_norm))
end

# load data
iris = load_iris()
iris = DataFrames.DataFrame(iris)
y, X = unpack(iris, ==(:target); rng=123);
X = Matrix(X)

x_transformed = PCA(X)
plot(x_transformed[:, 1], x_transformed[:, 2], seriestype=:scatter)