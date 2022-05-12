using Distributions
using Plots

d1 = Normal(1, 0.2)
d2 = Normal(3, 0.6)
d3 = Normal(5, 0.5)

x1 = rand(d1, 200)
x2 = rand(d2, 200)
x3 = rand(d3, 200)
x = vcat(x1, x2, x3)

histogram(x, bins=50)