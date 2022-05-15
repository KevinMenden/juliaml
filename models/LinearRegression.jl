# Linear Regression module

using LinearAlgebra
using Zygote
using Plots 
import Random

"""
Linear model

Args:
    x: input Vector
    ω: parameter vector
"""
function predict(x::Vector{<:Number}, ω::Vector{<:Number})
    mapreduce(j->ω[j] * x[j], (+), 1:size(x)[1], init=0)
end


"""
Loss function
"""
function loss(
    X::Matrix{<:Number},
    y::Vector{<:Number},
    ω::Vector{<:Number},
    λ::Real
    )::Number
    N = size(y)[1]
    1 // 2 * sum(map(n -> (y[n] - predict(X[n, :], ω))^2 + λ*norm(ω)^2 , 1:N))
end

"""
One step of gradient descent
"""
function gradient_descent_step(X, y, w)
    λ = 0.5
    lr = 0.001
    grads = gradient(w -> loss(X, y, w, λ), w)
    w - lr * grads[1]
end



X = [[1, 2, 2] [2, 2, 2] [3, 4, 1]]
y = [1, 3, 3.4]
ω = [1, 2, 3]

λ = 0.5
n_steps = 100


losses = []
for i = 1:n_steps
    l = loss(X, y, ω, λ)
    push!(losses, l)
    ω = gradient_descent_step(X, y, ω)

end

plot(1:n_steps, losses)