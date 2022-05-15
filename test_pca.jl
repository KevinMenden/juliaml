
include("models/PCA.jl")
import .pca


using MLJ
using Plots
import DataFrames
using MLDatasets
using UMAP: umap

train_x, train_y = MNIST.traindata()
n_samples = 10000
x_sub = train_x[:,:, 1:n_samples]
y_sub = train_y[1:n_samples]

x_sub = convert(Matrix{Float32} ,transpose(reshape(x_sub, (784, n_samples))))

# Test PCA
x_pca = pca.PCA(x_sub)
scatter(
    x_pca[:, 1],
    x_pca[:, 2],
    marker_z = y_sub, 
    alpha=0.5)


# Test UMAP
x_umap = umap(transpose(x_sub); n_neighbors=10, min_dist=0.001, n_epochs=200)
scatter(x_umap[1,:], x_umap[2,:], marker_z = y_sub)


