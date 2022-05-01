"""PCA implementation."""

using LinearAlgebra
using Random
using Statistics

X = randn(20, 5)
X_norm = X .- mean(X)

χ = cov(X_norm, dims=2)

evals = eigvals(χ)
evecs = eigvecs(χ)


sort_index = sortperm(evals, rev=true)
evals = evals[sort_index]

