include("AM_scaling_utils.jl")

dta_dict = stan_data("mRNA")
N = dta_dict["N"]
ts = dta_dict["ts"]
ys = dta_dict["ys"]

mRNA_target = StanLogPotential(
    joinpath(dirname(dirname(Base.active_project())), "stan", "mRNA.stan"), 
    Pigeons.json(; N, ts, ys)
)
prior_ref = DistributionLogPotential(product_distribution(
    Uniform(-2,1),Uniform(-5,5),Uniform(-0.25,5),Uniform(-5,-0.25),Uniform(-2,2)
))

pt, time, sample, n_leapfrog = pt_sample_from_model(
    mRNA_target, 1, AutoMALA(), nothing, 14; n_chains = 8,
    reference =prior_ref 
)

# using CairoMakie
# using PairPlots
# p = pairplot(get_component_samples(sample, 1:5))

# using StatsPlots
# p = corrplot(get_component_samples(sample, 1:5), size = (1200,800))
# savefig(p,"temp.png")
# println(get_component_samples(sample, 1:5)[end,:])

# dta_dict = stan_data("mRNA")
# time, sample, n_leapfrog = nuts_sample_from_model("mRNA", 1, 15)
