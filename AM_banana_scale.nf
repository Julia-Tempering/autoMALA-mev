include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    dim: (-13..6).collect{Math.pow(2, it)},
    seed: (1..20),
    model: ["banana_scale"],
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed", "NUTS"] // MALA runs alongside autoMALA 
]

model_string = [
    banana_scale: "Pigeons.stan_banana(1, 1/dim)", 
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

n_rounds = params.dryRun ? 4 : 20
PT_n_chains = 10
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
    memory {4.GB * task.attempt}
    time { 1.hour * task.attempt }
    errorStrategy 'retry'
    maxRetries '3'
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
    output:
        tuple val(arg), path('csvs')
  script:
    template 'AM_scale_main.jl'
}

