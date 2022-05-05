
include("PCA.jl")
import .pca


using MLJ
using Plots
import DataFrames
using MLDatasets


train_x, train_y = MNIST.traindata()
n_samples = 10000
x_sub = x[:, 1:n_samples]
y_sub = train_y[1:n_samples]
x_sub = convert(Matrix{Float32} ,transpose(x_sub))

# Test PCA
x_pca = pca.PCA(x_sub)
scatter(
    x_pca[:, 1],
    x_pca[:, 2],
    marker_z = y_sub, 
    alpha=0.5)

