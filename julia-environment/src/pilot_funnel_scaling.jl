using Pigeons
using Statistics
using StanSample
include("AM_scaling_utils.jl")

scale = 1/2.0
n_rounds = 18
dim = 1/scale # scale := 1/dim for these simulations 
model = "funnel_scale"

# pt = pigeons(
#     target = Pigeons.stan_funnel(1, scale), 
#     explorer = Pigeons.AutoMALA(),
#     record = [online], 
#     n_chains = 1, 
#     n_rounds = n_rounds)
# println(mean(pt)[1])
# println(var(pt)[1])

set_cmdstan_home!("/home/nik/Downloads/cmdstan/")
stan_model = model_string(model)
sm = SampleModel(model, stan_model) 
sm.num_threads = 1
sm.num_julia_chains = 1
sm.num_chains = 1
sm.num_samples = 2^n_rounds 
sm.num_warmups = 2^n_rounds - 2
sm.save_warmup = true
sm.seed = 1
data = stan_data(model, dim)
rc = stan_sample(sm; data)
sample = DataFrame(read_samples(sm))[(sm.num_warmups+1):end,:] # discard warmup
println(Statistics.mean(sample, 1))
println(Statistics.var(sample, 1))


# ----------- NUTS -----------
# scale     n_rounds 
# 1/0.1     15
# 1/0.2     15
# 1/0.3     19
# 1/0.4     21
# 1/0.5     >23 (maybe 24?, didn't test yet)
# 1/0.6     (maybe 25)
# 1/0.7     
# -----------------------------

# ---------- Pigeons ----------
# scale     n_rounds 
# 1/0.1     15
# 1/0.2     15
# 1/0.3     16
# 1/0.4     16
# 1/0.5     17
# 1/0.6     19
# 1/0.7     20
# ...
# 1/1.0     > 20 ..     
# -----------------------------



