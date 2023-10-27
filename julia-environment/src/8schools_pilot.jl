include("AM_scaling_utils.jl")
centered = false
target = Pigeons.stan_eight_schools(centered)

# get approximate mean,std for tau (just done once)
pt = pigeons(
    target      = target, 
    variational = GaussianReference(),
    seed        = 1,
    n_rounds    = 15,
    n_chains    = 4,
    explorer    = AutoMALA(), 
    show_report = true,
    record      = [traces]
)
samples = get_sample(pt) # compute summaries of last round (0.5 of all samples)
dim = length(first(samples))
taus = get_component_samples(samples, dim)
mean_and_std(taus)


model = "eight_schools_noncentered"
time, sample, n_leapfrog = nuts_sample_from_model(model, 1, 15)

