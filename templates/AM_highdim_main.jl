#!/usr/bin/env -S julia --heap-size-hint=${task.memory.toGiga()}G
using Pkg
Pkg.activate(joinpath("$baseDir", "$julia_env")) 
using Pigeons
using CSV
using DataFrames
using MCMCChains
using StanSample 
using HypothesisTests
using Distributions 
using Statistics
include(joinpath("$baseDir", "$julia_env", "src", "AM_scaling_utils.jl"))

function main()
	# collect global vars 
	scale = $scale
	explorer_type = "${arg.sampler}"
	explorer = ${sampler_string[arg.sampler]}
	dim = ${arg.dim}
	target = ${model_string[arg.model]}
	model = "${arg.model}"
	seed = ${arg.seed}
	n_rounds = $n_rounds

	if explorer_type != "NUTS" # use Pigeons 
	    pt, time, sample, n_leapfrog = pt_sample_from_model(target, seed, explorer, explorer_type, n_rounds)
	else # use cmdstan for NUTS
	    time, sample, n_leapfrog = nuts_sample_from_model(model, seed, n_rounds; dim=dim, scale=scale)     
	end
	ess = margin_ess(sample)
	ess_exact = margin_ess(sample, model)
	msjd = MSJD(sample)
	margin1_mean = mean(sample, 1) 
	margin1_var = var(sample, 1) 
	ks, pval = KS_statistic(sample, Normal(special_margin_mean_std(model)...), 1)
	df = DataFrame(
	    ess = ess, ess_exact = ess_exact, n_leapfrog = n_leapfrog, msjd = msjd, 
	    margin1_mean = margin1_mean, margin1_var = margin1_var, 
	    KS = ks, ks_pval = pval, mala = false)

	!isdir("csvs") ? mkdir("csvs") : nothing
	CSV.write("csvs/summary.csv", df)
end

main()

