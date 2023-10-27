include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    seed: (1..20),
    model: ["normal", "funnel", "banana"],
    dim: (1..1).collect{pow(2, it)},
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed"] // MALA runs after autoMALA
]

def model_string = [
    funnel: "Pigeons.stan_funnel(dim-1, 2.0)", // NB: funnel and banana have extra parameter
    banana: "Pigeons.stan_banana(dim-1)",
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
    memory '4 GB' // keep in sync with the heap-size-hint arg below
    time '35min'
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

        pt, time, sample, n_leapfrog = pt_sample_from_model(target, seed, explorer, explorer_type, n_rounds)
        ess = margin_ess(sample)
        ess_exact = margin_ess(sample, model)
        msjd = MSJD(sample)
        base_step_size = get_step_size(pt.shared.explorer)
        acceptance = Pigeons.explorer_mh_prs(pt)[1]
        df = DataFrame(
            ess = ess, ess_exact = ess_exact, n_leapfrog = n_leapfrog, msjd = msjd, mala = false, 
            stepsize = base_step_size, acceptance = acceptance)

        explorer_type = "MALA"
        step_sizes = [2. .^(-6:-1); 2. .^(0:3)] * base_step_size
        for step_size in step_sizes
            explorer = ${sampler_string["MALA"]}
            pt_mala, time_mala, sample_mala, n_leapfrog_mala = pt_sample_from_model(target, seed, explorer, explorer_type, n_rounds)
            ess_mala = margin_ess(sample_mala)
            ess_exact_mala = margin_ess(sample_mala, model)
            msjd_mala = MSJD(sample_mala)
            acceptance_mala = Pigeons.explorer_mh_prs(pt_mala)[1]
            df_mala = DataFrame(
                ess = ess_mala, ess_exact = ess_exact_mala, n_leapfrog = n_leapfrog_mala, msjd = msjd_mala, 
                mala = true, stepsize = step_size, acceptance = acceptance_mala)
            df = vcat(df, df_mala)
        end

        !isdir("csvs") ? mkdir("csvs") : nothing
        CSV.write("csvs/summary.csv", df)
    end 

    main()
    """
}


process plot {
    input:
        path aggregated
    """ 
    #!/usr/bin/env julia  
    
    using CSV 
    using DataFrames 
    using Pigeons
    using Plots
    using Statistics
    include(joinpath("$baseDir", "$julia_env_dir", "src", "AM_scaling_utils.jl"))

    df = DataFrame(CSV.File("${aggregated}/summary.csv")) 
    path = "${aggregated}/../"
    mala_stepsize_performance_plot(df, path)
    """
}
