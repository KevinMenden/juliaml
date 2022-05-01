
"""PCA implementation."""

using LinearAlgebra
using Random
using Statistics
using MLJ
using Plots
import DataFrames

# load data
iris = load_iris()
iris = DataFrames.DataFrame(iris)
y, X = unpack(iris, ==(:target); rng=123);
X = Matrix(X)

# centering the data
X_norm = X .- mean(X)

# covariance
χ = cov(X_norm, dims=1)

# eigenvectors and values
evals = eigvals(χ)
evecs = eigvecs(χ)

sort_index = sortperm(evals, rev=true)
evecs = evecs[:, sort_index]

n_components = 2
evecs_sub = evecs[:, 1:n_components]

x_transformed = transpose(transpose(evecs_sub) * transpose(X_norm))

plot(x_transformed[:, 1], x_transformed[:, 2], seriestype=:scatter)