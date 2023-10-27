include { crossProduct; collectCSVs; setupPigeons; head; pow; deliverables; checkGitUpdated; commit } from './utils.nf'
params.dryRun = false

def variables = [
    dim: (0..8),
    seed: (1..20),
    model: ["two_component_normal_scale"],
    sampler: ["mix_autoMALA","autoMALA", "autoMALA_fixed", "NUTS"] // MALA runs alongside autoMALA 
]

model_string = [
    two_component_normal_scale: "make_2_comp_norm_target(1, dim)", // 2d normal, dim interpreted as exponent for stdevs
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

PT_n_chains = 10
n_rounds = params.dryRun ? 4 : 20 
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
    memory '4G'
    time '1h'
//    errorStrategy 'retry'
//    maxRetries '3'
    input:
        env JULIA_DEPOT_PATH
        path julia_env
        val arg
    output:
        tuple val(arg), path('csvs')
  script:
    template 'AM_scale_main.jl'
}

