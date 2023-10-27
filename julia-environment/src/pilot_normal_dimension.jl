using Pigeons
using Statistics
using StanSample
include("AM_scaling_utils.jl")

n_rounds = 10
dim = 1024
model = "normal"

pt = pigeons(
    target = Pigeons.ScaledPrecisionNormalPath(1.0, 1.0, dim),
    explorer = Pigeons.AutoMALA(),
    record = [online], 
    n_chains = 1, 
    n_rounds = n_rounds
)
println(mean(pt)[1])
println(var(pt)[1])


# set_cmdstan_home!("/home/nik/Downloads/cmdstan/")
# stan_model = model_string(model)
# sm = SampleModel(model, stan_model) 
# sm.num_threads = 1
# sm.num_julia_chains = 1
# sm.num_chains = 1
# sm.num_samples = 2^n_rounds 
# sm.num_warmups = 2^n_rounds - 2
# sm.save_warmup = true
# sm.seed = 1
# data = stan_data(model, dim, scale) 
# rc = stan_sample(sm; data)
# sample = DataFrame(read_samples(sm))[(sm.num_warmups+1):end,:] # discard warmup
# println(Statistics.mean(sample, 1))
# println(Statistics.var(sample, 1))



# ----------- NUTS -----------
# dim       scale       n_rounds    outcome 
# -----------------------------

# ---------- Pigeons ----------
# dim       scale       n_rounds 
# -----------------------------



