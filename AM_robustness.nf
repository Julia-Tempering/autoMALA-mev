include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    dim: (1..1).collect{pow(2, it)},
    seed: (1..20),
    model: ["funnel", "banana", "normal"],
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed", "MALA"],
    step_size: (-10.0..1.0).collect{java.lang.Math.pow(2.0, it)}
]

def model_string = [
    funnel: "Pigeons.stan_funnel(dim-1, 2.0)",
    banana: "Pigeons.stan_banana(dim-1)",
    normal: "Pigeons.ScaledPrecisionNormalPath(1.0, 1.0, dim)"
]

def sampler_string = [
    autoMALA: "AutoMALA(step_size = step_size)",
    autoMALA_fixed: "AutoMALA(step_size = step_size, preconditioner = Pigeons.DiagonalPreconditioner())",
    mix_autoMALA: """Mix(
        AutoMALA(step_size = step_size, preconditioner=Pigeons.IdentityPreconditioner(), base_n_refresh=1),
        AutoMALA(step_size = step_size, preconditioner=Pigeons.MixDiagonalPreconditioner(), base_n_refresh=1),
        AutoMALA(step_size = step_size, preconditioner=Pigeons.DiagonalPreconditioner(), base_n_refresh=1)
    )""",
    MALA: "Pigeons.MALA(step_size = step_size)", 
]

def n_rounds = params.dryRun ? 4 : 18
def julia_env_dir = file("julia-environment")
def julia_depot_dir = file(".depot")
def deliv = deliverables(workflow)

workflow {
    args = crossProduct(variables, params.dryRun)
    julia_env = setupPigeons(julia_depot_dir, julia_env_dir)
    agg_path = runSimulation(julia_depot_dir, julia_env, args) | collectCSVs 
    //commit(agg_path, params.dryRun) // cannot commit from container, priv keys not available
}

process runSimulation {
    memory '4GB'
    time '15min'
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
    output:
        tuple val(arg), path('csvs')
    
    """ 
    #!/usr/bin/env -S julia --heap-size-hint=${task.memory.toGiga()}G
    using Pkg
    Pkg.activate(joinpath("$baseDir", "$julia_env")) 

    using Pigeons
    using CSV
    using DataFrames
    using MCMCChains
    using StanSample 
    include(joinpath("$baseDir", "$julia_env", "src", "AM_scaling_utils.jl"))

    # collect global vars 
    step_size = Float64(${arg.step_size})
    explorer_type = "${arg.sampler}"
    explorer = ${sampler_string[arg.sampler]}
    dim = ${arg.dim}
    target = ${model_string[arg.model]}
    model = "${arg.model}"
    seed = ${arg.seed}
    n_rounds = $n_rounds

    pt, time, sample, n_leapfrog = pt_sample_from_model(target, seed, explorer, explorer_type, n_rounds)
    ess = margin_ess(sample)
	ess_exact = margin_ess(sample, model)
    msjd = MSJD(sample)
    acceptance = Pigeons.explorer_mh_prs(pt)[1]
    final_step_size = get_step_size(pt.shared.explorer)
    df = DataFrame(
        ess = ess, ess_exact = ess_exact, n_leapfrog = n_leapfrog, msjd = msjd, 
        acceptance = acceptance, final_step_size = final_step_size)
    !isdir("csvs") ? mkdir("csvs") : nothing
    CSV.write("csvs/summary.csv", df)
    """
}
