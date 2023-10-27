/* 
For assessing: 
1. Scaling of automatic step size selection in autoMALA with respect to dimension
2. Convergence of autoMALA step size as n_round increases
*/

include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    dim: (1..9).collect{pow(2, it)},
    seed: (1..20),
    model: ["funnel", "banana", "normal"],
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed"] 
]

def model_string = [
    funnel: "target = Pigeons.stan_funnel(dim-1)", // NB: funnel and banana have one extra parameter
    banana: "target = Pigeons.stan_banana(dim-1)",
    normal: "target = Pigeons.ScaledPrecisionNormalPath(1.0, 1.0, dim)"
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

def n_rounds = params.dryRun ? 4 : 18 // 18 seems OK for now, max #rounds (collect data at each previous round)
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
    memory '16 GB'
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
    using DataFrames 
    using CSV
    get_step_size(explorer) = explorer.step_size
    get_step_size(explorer::Mix) = first(explorer.explorers).step_size
    function main()
        step_sizes = Vector{Float64}(undef, $n_rounds)
        dim = ${arg.dim}
        pt = pigeons( 
            ${model_string[arg.model]},
            seed        = ${arg.seed},
            n_rounds    = 1,
            n_chains    = 1, 
            record      = record_default(),
            explorer    = ${sampler_string[arg.sampler]}, 
            show_report = false
        )

        step_sizes[1] = get_step_size(pt.shared.explorer)
        for i in 2:$n_rounds
            pt = Pigeons.increment_n_rounds!(pt, 1)
            pt = pigeons(pt)
            step_sizes[i] = get_step_size(pt.shared.explorer)
        end 

        df = DataFrame(round = 1:$n_rounds, step_size = step_sizes)
        !isdir("csvs") ? mkdir("csvs") : nothing
        CSV.write("csvs/summary.csv", df)
    end 

    main()
    """
}


process plot {
    input:
        path aggregated
    output:
        path '*.*'
    publishDir deliv, mode: 'copy', overwrite: true
    """ 
    #!/usr/bin/env Rscript
    require("ggplot2")
    require("dplyr")

    # TODO: see the Julia plotting scripts 
    """
}
