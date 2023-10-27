include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    dim: [10, 100, 1000], // interpreted as (maximum) n_obs, actual dim=n_predictor is fixed
    seed: (1..20),
    model: ["horseshoe"],
    dataset: ["prostate_small", "ionosphere", "sonar"],
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed", "NUTS"] // MALA runs alongside autoMALA 
]

model_string = [
    horseshoe: "make_HSP_target(dataset,dim)" 
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

n_rounds = params.dryRun ? 4 : 14
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
    cpus { arg.sampler == 'PT' ? (params.dryRun ? 2 : 8) : 1 }
    memory { 1.GB * Math.max(4.0, 1.0*task.cpus) * task.attempt }
    time { 1.hour * task.attempt }
    errorStrategy 'retry'
    maxRetries '8'
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
    output:
        tuple val(arg), path('csvs')
  script:
    template 'AM_horseshoe_main.jl'
}

