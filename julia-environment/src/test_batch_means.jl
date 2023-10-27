include("batch_means.jl")

n_sample = 100_000
σ = 1.0 
μ = 0.0
sample = σ*randn(n_sample) .+ μ

println(batch_means_ess(sample, μ, σ))