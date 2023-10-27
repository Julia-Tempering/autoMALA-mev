#=
Batch means ESS with known mean and stdev
Implementation based on
https://github.com/UBC-Stat-ML/bayonet/blob/d751ebf27607806527481a1626ee75ae0d735a46/src/main/java/bayonet/math/EffectiveSampleSize.java

Basics of Batch Means

We want
    s^2 := lim_{n\to\infty} n*var(mean_{1:n}(x_i-E[x]))
Note
    mean_{1:n}(x_i-E[x]) = (1/n) sum_{1:n}(x_i-E[x])
    = (1/n) sum_{b=1}^n_batch sum_{a=1}^batch_size (x_{batch_size*(b-1)+a}-E[x])
    = (batch_size/n) sum_{b=1}^n_batch centered_batch_means[b]
    = (batch_size*n_batch/n) mean_{1:n_batch}(centered_batch_means)
    ≈ mean_{1:n_batch}(centered_batch_means) // exact for n_samples = n^2 for some n
so
    var(mean_{1:n}(x_i-E[x])) = var(mean_{1:n_batch}(centered_batch_means))
    ≈ (1/n_batch)var(centered_batch_means)
if centered_batch_means are roughly iid. Then, ess is approximately
    ESS ≈ n_batch * posterior-var(x)/var(centered_batch_means)
        = n_batch / var(centered_batch_means/posterior-sd(x))
and the standardization of the last step can be numercally more stable if done
inside the loop where centered_batch_means is computed
=#
using Statistics
using StatsBase

batch_means_ess(samples::AbstractVector) = 
    batch_means_ess(samples, mean_and_std(samples)...)
function batch_means_ess(
    samples::AbstractVector,
    posterior_mean::Real,
    posterior_sd::Real
)
    n_samples = length(samples)
    n_blocks  = 1 + isqrt(n_samples)
    blk_size  = n_samples ÷ n_blocks # takes floor of division
    centered_batch_means = map(1:n_blocks) do b
        i_start = blk_size*(b-1) + 1
        i_end   = blk_size*b
        mean(x -> (x - posterior_mean)/posterior_sd, @view samples[i_start:i_end])
    end
    n_blocks / mean(abs2, centered_batch_means)
end
