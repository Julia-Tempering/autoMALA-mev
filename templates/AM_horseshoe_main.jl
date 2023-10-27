#!/usr/bin/env -S julia --heap-size-hint=${task.memory.toGiga()}G
using Pkg
Pkg.activate(joinpath("$baseDir", "$julia_env")) 
include(joinpath("$baseDir", "$julia_env", "src", "AM_scaling_utils.jl")) # loads dependencies too

function main()
	# collect global vars 
	explorer_type = "${arg.sampler}"
	explorer = ${sampler_string[arg.sampler]}
    dim = ${arg.dim} # interpreted as n_obs, actual dim=n_predictor is fixed
	dataset = "${arg.dataset}"
	target = ${model_string[arg.model]}
	model = "${arg.model}"
	seed = ${arg.seed}
	n_rounds = ${n_rounds}
	n_chains = ${task.cpus}

	if explorer_type == "PT" # run PT with a variational reference (and on ChildProcesses if n_chains>1)
		n_chains==1 && error("Using n_chains=1 while attempting to run variationalPT")
		r = pigeons(
			target      = target, 
			seed        = seed,
			n_rounds    = n_rounds,
			n_chains    = n_chains, 
			explorer    = AutoMALA(),
			checkpoint  = true,
			variational = GaussianReference(), 
			show_report = true,
			record      = push!(record_default(),traces),
		        on = ChildProcess(n_local_mpi_processes = n_chains))
		sample = get_sample(load(r))
		n_leapfrog = NaN
	elseif explorer_type == "NUTS" # use cmdstan for NUTS
		time, sample, n_leapfrog = nuts_sample_from_model(model, seed, n_rounds; dim=dim,dataset=dataset)
	else # use Pigeons with 1 chain
	    pt, time, sample, n_leapfrog = pt_sample_from_model(target, seed, explorer, explorer_type, n_rounds)
	end
	ess_batch  = min_ess_batch(sample)
	ess_chains = min_ess_chains(sample)
	df = DataFrame(
	    ess = ess_batch, ess_exact = ess_chains, # keep convention to not break scripts
		n_leapfrog = n_leapfrog, msjd = NaN, 
		margin1_mean = NaN, margin1_var = NaN, 
	    KS = NaN, ks_pval = NaN, mala = false)

	!isdir("csvs") ? mkdir("csvs") : nothing
	CSV.write("csvs/summary.csv", df)
end

main()

