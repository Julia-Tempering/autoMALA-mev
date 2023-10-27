#!/usr/bin/env -S julia --heap-size-hint=${task.memory.toGiga()}G
using Pkg
Pkg.activate(joinpath("$baseDir", "$julia_env")) 
include(joinpath("$baseDir", "$julia_env", "src", "AM_scaling_utils.jl")) # loads dependencies too

function main()
	# collect global vars 
	explorer_type = "${arg.sampler}"
	explorer = ${sampler_string[arg.sampler]}
	dim = ${arg.dim} # interpreted as scale
	target = ${model_string[arg.model]}
	model = "${arg.model}"
	seed = ${arg.seed}
	n_rounds = ${n_rounds}
	model == "funnel_scale" && occursin("autoMALA", explorer_type) && (n_rounds -= 1) # equalize efforts

	if explorer_type != "NUTS" # use Pigeons 
	    (explorer_type != "PT") ? (n_chains = 1) : n_chains = $PT_n_chains
	    pt, time, sample, n_leapfrog = pt_sample_from_model(target, seed, explorer, explorer_type, n_rounds; n_chains = n_chains)
	else # use cmdstan for NUTS
	    time, sample, n_leapfrog = nuts_sample_from_model(model, seed, n_rounds; dim=dim)     
	end
	idx_margin = special_margin_idx(model,n_vars(sample))
	ess = margin_ess(sample)
	ess_exact = margin_ess(sample, model, idx_margin, dim) # dim is passed to special_margin_mean_std
	msjd = MSJD(sample)
	margin1_mean = mean(sample, idx_margin) 
	margin1_var = var(sample, idx_margin) 
	ks, pval = KS_statistic(sample, Normal(special_margin_mean_std(model, dim)...), idx_margin)
	df = DataFrame(
	    ess = ess, ess_exact = ess_exact, n_leapfrog = n_leapfrog, msjd = msjd, 
	    margin1_mean = margin1_mean, margin1_var = margin1_var, 
	    KS = ks, ks_pval = pval, mala = false)

	!isdir("csvs") ? mkdir("csvs") : nothing
	CSV.write("csvs/summary.csv", df)
end

main()

