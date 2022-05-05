
"""PCA implementation."""

module pca
using LinearAlgebra
using Random
using Statistics

function PCA(X::Matrix{<:Number}, n_components::Integer=2)
    X_norm = X .- mean(X, dims=2)
    χ = cov(X_norm, dims=1)
    evals = eigvals(χ)
    evecs = eigvecs(χ)

    sort_index = sortperm(evals, rev=true)
    evecs = evecs[:, sort_index]
    evecs_sub = evecs[:, 1:n_components]
    transpose(transpose(evecs_sub) * transpose(X_norm))
end

end