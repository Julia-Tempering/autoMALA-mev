include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    seed: (1..20),
    model: ["normal"],
    dim: (1..14).collect{pow(2, it)},
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed"] // MALA runs alongside autoMALA 
]

def model_string = [
    normal: "Pigeons.ScaledPrecisionNormalPath(1.0, 1.0, dim)"
]

sampler_string = [ 
    autoMALA: "AutoMALA()",
    autoMALA_fixed: "AutoMALA(preconditioner = Pigeons.DiagonalPreconditioner())",
    mix_autoMALA: """Mix(
        AutoMALA(preconditioner=Pigeons.IdentityPreconditioner(), base_n_refresh=1),
        AutoMALA(preconditioner=Pigeons.MixDiagonalPreconditioner(), base_n_refresh=1),
        AutoMALA(preconditioner=Pigeons.DiagonalPreconditioner(), base_n_refresh=1)
    )""", 
]

def n_rounds = params.dryRun ? 4 : 15
def julia_env_dir = file("julia-environment")
def julia_depot_dir = file(".depot")
def deliv = deliverables(workflow)

workflow {
    args = crossProduct(variables, params.dryRun)
    //julia_env = setupPigeons(julia_depot_dir, julia_env_dir)
    agg_path = runSimulation(julia_depot_dir, julia_env_dir, args) | collectCSVs 
    //commit(agg_path, params.dryRun) // cannot commit from container, priv keys not available
}

process runSimulation {
    memory '8GB'
    time '2h'
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

    function main()
        # collect global vars 
        explorer_type = "${arg.sampler}"
        explorer = ${sampler_string[arg.sampler]}
        dim = ${arg.dim}
        target = ${model_string[arg.model]}
        model = "${arg.model}"
        seed = ${arg.seed}
        n_rounds = $n_rounds

        pt, time, sample, n_leapfrog = pt_sample_from_model(target, seed, explorer, explorer_type, n_rounds; keep_traces = false)
        acceptance = Pigeons.explorer_mh_prs(pt)[1]
        df = DataFrame(acceptance = acceptance)
        !isdir("csvs") ? mkdir("csvs") : nothing
        CSV.write("csvs/summary.csv", df)
    end 

    main()
    """
}
