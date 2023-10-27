include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    dim: (1..10).collect{pow(2, it)},
    seed: (1..20),
    model: ["banana"],
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed", "NUTS"] // MALA runs alongside autoMALA 
]

model_string = [
    funnel: "Pigeons.stan_funnel(dim-1, scale)", // NB: funnel and banana have extra parameter
    banana: "Pigeons.stan_banana(dim-1, scale)",
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
    NUTS: "Pigeons.MALA()", // ignored, just use it to compile
]

scale = 1.0
n_rounds = params.dryRun ? 4 : 18
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
    // linearly scale mem with dim*(total number of samples when doing n_rounds = 2^(n_rounds+1)-2 ~ 2^(n_rounds+1))
    // reference is 6G for 128 dim and 18 rounds
    memory { 1.GB * Math.round(
        Math.min(92.0,     // smallest machines on beluga
            Math.max(4.0,   // ~ fixed mem cost
                task.attempt * 6.0 * (arg.model == "funnel_scale" ? 2 : arg.dim) * Math.pow(2,n_rounds+1)  / (128.0 * 524288)
            )
        )
    ) }
    time '4h'
    errorStrategy 'retry'
    maxRetries '3'
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
    output:
        tuple val(arg), path('csvs')
    script: 
        template 'AM_highdim_main.jl'
}
