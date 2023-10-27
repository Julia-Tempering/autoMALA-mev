using Colors
using CSV 
using DataFrames
using Dates
using Distributions
using HypothesisTests
using MCMCChains
using Pigeons
using Plots
using Plots.PlotMeasures: px
using Random
using SplittableRandoms: SplittableRandom
using Statistics
using StanSample
using StatsPlots

import Pigeons: initialization, sample_iid!

###############################################################################
# ESS and friends
###############################################################################

include("batch_means.jl")

const PigeonsSample = Union{Pigeons.SampleArray{<:AbstractVector}, Vector{<:AbstractVector}}

get_component_samples(samples::PigeonsSample, idx_component) =
    collect(hcat((s[idx_component] for s in samples)...)')
get_component_samples(samples::PigeonsSample, idx_component::Int) =
    [s[idx_component] for s in samples]
get_component_samples(samples::DataFrame, idx_component) = Array(samples[:,idx_component])

# ESS
function margin_ess(samples, model_exact_means=nothing, margin_idx=1, args...)
    margin = get_component_samples(samples,margin_idx)
    isnothing(model_exact_means) && return batch_means_ess(margin)
    batch_means_ess(margin, special_margin_mean_std(model_exact_means, args...)...) 
end
special_margin_idx(model::AbstractString, dim::Int) = 
    startswith(model, "two_component_normal") ? dim : one(dim)

special_margin_mean_std(model::String, args...) = 
    if startswith(model,"funnel")
        (0.,3.)
    elseif startswith(model,"banana")
        (0., sqrt(10))
    elseif startswith(model,"eight_schools")    # approx using variational PT (see 8schools_pilot.jl)
        (3.574118538746056, 3.1726880307401455)
    elseif startswith(model, "normal") 
        (0., 1.)
    elseif startswith(model, "two_component_normal")
        (0., last(two_component_normal_stdevs(args...))) # the margin with the largest stdev 
    else
        error("unknown model $model")
    end
two_component_normal_stdevs(e::Real,args...) = (10. ^(-e), 10. ^(e))
n_vars(samples::PigeonsSample) = length(first(samples))
n_vars(samples::DataFrame) = size(samples,2)
min_ess_batch(samples) = minimum(1:n_vars(samples)) do i
        batch_means_ess(get_component_samples(samples, i))
    end
to_chains(samples::PigeonsSample) = Chains(samples)
to_chains(samples::DataFrame) = Chains(Array(samples))
min_ess_chains(samples) = minimum(ess(to_chains(samples),kind=:basic).nt.ess) # :basic is actual ess of the vars. default is :bulk which computes ess on rank-normalized vars

# TODO: should we discard the warmup samples?
function MSJD(sample::PigeonsSample) 
    n = length(sample) 
    msjd = 0.0
    for i in 2:n 
       msjd += sum((sample[i] .- sample[i-1]) .^ 2) / (n-1)
    end 
    return msjd
end 

function MSJD(sample::DataFrame) 
    sample_vec = df_to_vec(sample) 
    return MSJD(sample_vec) 
end 

function Statistics.mean(sample::PigeonsSample, margin)
    x = [sample[i][margin] for i in eachindex(sample)]
    return mean(x) 
end

function Statistics.mean(sample::DataFrame, margin)
    sample_vec = df_to_vec(sample)
    return mean(sample_vec, margin)
end

function Statistics.var(sample::PigeonsSample, margin)
    x = [sample[i][margin] for i in eachindex(sample)]
    return var(x) 
end

function Statistics.var(sample::DataFrame, margin)
    sample_vec = df_to_vec(sample)
    return var(sample_vec, margin)
end

#=
Kolmogorov-Smirnov test.  
=#
function KS_statistic(sample::PigeonsSample, d::Distribution, margin = 1)
    x = [sample[i][margin] for i in 1:length(sample)]
    t = HypothesisTests.ApproximateOneSampleKSTest(x, d)
    return sqrt(t.n)*t.δ, pvalue(t) 
end 

function KS_statistic(sample::DataFrame, d::Distribution, margin = 1) 
    sample_vec = df_to_vec(sample) 
    return KS_statistic(sample_vec, d, margin) 
end

function df_to_vec(df::DataFrame) 
    n = size(df,1)
    df_vec = Vector{Vector{Float64}}(undef, n) 
    for i in 1:n 
        df_vec[i] = Vector(df[i, :]) 
    end 
    return df_vec
end 

###############################################################################
# sampling
###############################################################################

function pt_sample_from_model(
    target, seed, explorer, explorer_type, args...;
    n_chains = 1, kwargs...
    )
    inp = Inputs(
        target      = target, 
        seed        = seed,
        n_rounds    = 1,
        n_chains    = n_chains, 
        explorer    = explorer, 
        show_report = true
    )
    pt_sample_from_model(inp, args...; kwargs...)
end

function pt_sample_from_model(inp::Inputs, n_rounds; keep_traces=true)
    # build pt
    recorders = [record_default(); Pigeons.explorer_acceptance_pr] 
    keep_traces && (recorders = vcat(recorders, Pigeons.traces))
    inp.record = recorders
    pt = PT(inp)

    # iterate rounds
    n_leapfrog = 0
    for _ in 1:n_rounds
        pt = pigeons(pt)
        n_leapfrog += Pigeons.explorer_n_steps(pt)[1]
        pt = Pigeons.increment_n_rounds!(pt, 1)
    end
    time   = sum(pt.shared.reports.summary.last_round_max_time)
    if keep_traces 
        sample = get_sample(pt) # compute summaries of last round (0.5 of all samples)
        @assert length(sample) == 2^n_rounds
    else 
        sample = nothing 
    end
    return pt, time, sample, n_leapfrog
end

function nuts_sample_from_model(model, seed, n_rounds; kwargs...)
    stan_model = model_string(model; kwargs...)
    sm = SampleModel(model, stan_model) 
    sm.num_threads      = 1
    sm.num_julia_chains = 1
    sm.num_chains       = 1
    sm.num_samples      = 2^n_rounds 
    sm.num_warmups      = 2^n_rounds - 2
    sm.save_warmup      = true
    sm.seed             = seed
    sm.show_logging     = true

    data = stan_data(model; kwargs...)
    time = @elapsed begin 
        rc = stan_sample(sm; data)
    end
    sample = DataFrame(read_samples(sm))[(sm.num_warmups+1):end,:] # discard warmup
    @assert nrow(sample) == 2^n_rounds == sm.num_samples "nrow(sample) = $(nrow(sample)), sm.num_samples=$(sm.num_samples)"
    info = DataFrame(CSV.File(joinpath(sm.tmpdir, model * "_chain_1.csv"), comment = "#"))
    @assert size(info,1) == sm.num_samples + sm.num_warmups
    n_leapfrog = sum(info.n_leapfrog__) # count leapfrogs during warmup
    return time, sample, n_leapfrog
end

###############################################################################
# plotting
###############################################################################

function banana_contours_plot()
    x = range(-20, 20, length=100)
    y = range(-200, 600, length=100)
    f(x, y) = abs2(x)/20 + abs2(y-abs2(x))/(2*var_y) # p(x,y) = N(x;0,sqrt(10)^2)N(y;x^2,s^2)
    var_y = 1e-1
    z = @. f(x', y)
    levs = 10. .^ range(log10.(quantile(z,[0.1,0.9]))..., length=10)
    contour(x, y, z,levels=levs)
    plot!(abs2,x,linestyle=:dash,linewidth=2,label="x²")
end


function AM_scaling_plot(df, path) 
    if df.model[1] == "banana_scale" 
        df.dim = 1.0 ./ df.dim
    end
    add_mala_to_df!(df) # convert autoMALA to MALA when mala = true
    sort!(df)
    samplers = unique(df.sampler)
    seeds = vcat(unique(df.seed), :mean)
    for model in unique(df.model)
        p = Plots.plot()
        p2 = Plots.plot()
        p3 = Plots.plot()
        p4 = Plots.plot()
        p5 = Plots.plot()
        p6 = Plots.plot()
        p7 = Plots.plot()
        p8 = Plots.plot()
        p9 = Plots.plot()
        p10 = Plots.plot()
        if model in ["funnel_scale", "banana_scale"]
            xlabel = "Inverse scale" 
        else 
            xlabel = "Dimension"
        end
        for seed in seeds
            for sampler in samplers
                df_subset = copy(df)
                if seed != :mean 
                    df_subset = filter(:seed => n -> n == seed, df_subset)
                end
                df_subset = filter(:model => n -> n == model, df_subset)
                seed != :mean ? alpha = 0.25 : alpha = 1.0
                df_subset = filter(:sampler => n -> n == sampler, df_subset)
                if seed == :mean 
                    df_subset = groupby(df_subset, :dim) # group by dimension and take average across seeds
                    df_subset = combine(
                        df_subset, :ess => mean, :ess_exact => mean, :n_leapfrog => mean, :msjd => mean, 
                        :margin1_mean => mean, :margin1_var => mean, :KS => mean, 
                        :ks_pval => mean, renamecols = false)
                end
                color = findfirst(samplers .== sampler)
                Plots.plot!(
                    p, df_subset.dim, df_subset.ess, 
                    xlabel = xlabel, ylabel = "ESS", 
                    xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha, 
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color 
                )
                Plots.plot!( 
                    p2, df_subset.dim, df_subset.msjd, 
                    xlabel = xlabel, ylabel = "MSJD", 
                    xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha,
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color
                )
                Plots.plot!( 
                    p3, df_subset.dim, df_subset.margin1_mean, 
                    xlabel = xlabel, ylabel = "mean(first margin)", xaxis = :log10, 
                    label = seed == :mean ? sampler : "", alpha = alpha,
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color
                )
                Plots.plot!( 
                    p4, df_subset.dim, df_subset.margin1_var, 
                    xlabel = xlabel, ylabel = "var(first margin)", xaxis = :log10, 
                    label = seed == :mean ? sampler : "", alpha = alpha,
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color
                )
                Plots.plot!( 
                    p5, df_subset.dim, df_subset.KS, 
                    xlabel = xlabel, ylabel = "KS statistic", xaxis = :log10, 
                    label = seed == :mean ? sampler : "", alpha = alpha,
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color
                )
                Plots.plot!( 
                    p6, df_subset.dim, df_subset.n_leapfrog ./ df_subset.ess, 
                    xlabel = xlabel, ylabel = "Leapfrogs per ESS", 
                    yaxis = :log10, xaxis = :log10, 
                    label = seed == :mean ? sampler : "", alpha = alpha,
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color
                )
                Plots.plot!( 
                    p7, df_subset.dim, df_subset.n_leapfrog ./ df_subset.ess_exact, 
                    xlabel = xlabel, ylabel = "Leapfrogs per ESS(μ, σ)", 
                    yaxis = :log10, xaxis = :log10,
                    label = seed == :mean ? sampler : "", alpha = alpha,
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color 
                )
                Plots.plot!(
                    p8, df_subset.dim, df_subset.ess_exact, 
                    xlabel = xlabel, ylabel = "ESS(μ, σ)", 
                    xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha, 
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color
                )
                Plots.plot!(
                    p9, df_subset.dim, min.(df_subset.ess, df_subset.ess_exact), 
                    xlabel = xlabel, ylabel = "minESS", 
                    xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha, 
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color
                )
                Plots.plot!( 
                    p10, df_subset.dim, df_subset.n_leapfrog ./ min.(df_subset.ess, df_subset.ess_exact), 
                    xlabel = xlabel, ylabel = "Leapfrogs per minESS", 
                    yaxis = :log10, xaxis = :log10,
                    label = seed == :mean ? sampler : "", alpha = alpha,
                    seriestype = seed == :mean ? :line : :scatter, color = color,
                    lc = color 
                )
            end 
        end
        savefig(p, path * "scaling-ess-model-" * model * ".pdf")
        savefig(p2, path * "scaling-msjd-model-" * model * ".pdf")
        savefig(p3, path * "scaling-margin1_mean-model-" * model * ".pdf")
        savefig(p4, path * "scaling-margin1_var-model-" * model * ".pdf")
        savefig(p5, path * "scaling-KS-model-" * model * ".pdf")
        savefig(p6, path * "scaling-leapfrog-model-" * model * ".pdf")
        savefig(p7, path * "scaling-leapfrog_exact-model-" * model * ".pdf")
        savefig(p8, path * "scaling-ess_exact-model-" * model * ".pdf")
        savefig(p9, path * "scaling-miness-model-" * model * ".pdf")
        savefig(p10, path * "scaling-leapfrog_min-model-" * model * ".pdf")
    end
end

function base_dir()
    base_folder = dirname(dirname(Base.active_project()))
    endswith(base_folder, "autoMALA-mev") || error("please activate the AM mev julia-environment")
    return base_folder
end

function get_summary_df(experiment::String)
    base_folder = base_dir()
    csv_path    = joinpath(base_folder, "deliverables", experiment, "aggregated", "summary.csv")
    df = DataFrame(CSV.File(csv_path))
    if experiment == "AM_banana_scale" && 
        Date(readchomp(`git log -1 --pretty="format:%ci" $csv_path`)[1:10], "yyyy-mm-dd") <= Date("2023-10-11", "yyyy-mm-dd") # check if it uses the inverse convention
        df.dim .= inv.(df.dim) # fix wrong dim convention
    end
    return df
end

get_step_size(explorer) = explorer.step_size
get_step_size(explorer::Mix) = first(explorer.explorers).step_size

function filter_single_AM_version!(df,AM_version)
    filter!(:sampler => (s -> !occursin("autoMALA",s) || s == AM_version), df)
    map!(s -> (s == AM_version ? "autoMALA" : s), df[!, :sampler], df[!, :sampler])
    return df
end

function AM_stepsize_scaling_plot(df,mode="") 
    sort!(df)
    seeds = vcat(unique(df.seed), :mean)
    models = unique(df.model)
    base_path = joinpath(
        base_dir(), "deliverables", "AM_stepsize_scaling" * mode)
    # separate plot for each dimension: step size vs tuning round (overlay three models)
    for dim in unique(df.dim)
        p = Plots.plot()
        for model in models 
            for seed in seeds
                df_subset = copy(df)
                if seed != :mean 
                    df_subset = filter(:seed => n -> n == seed, df_subset)
                end
                df_subset = filter(:model => n -> n == model, df_subset)
                df_subset = filter(:dim => n -> n == dim, df_subset)
                seed != :mean ? alpha = 0.25 : alpha = 1.0
                if seed == :mean 
                    df_subset = groupby(df_subset, :round) # group by round and take average across seeds
                    df_subset = combine(df_subset, :step_size => mean, renamecols = false)
                end
                color = findfirst(models .== model)
                Plots.plot!(
                    p, df_subset.round, df_subset.step_size, 
                    xlabel = "Tuning round", ylabel = "Step size", 
                    label = seed == :mean ? model : "", alpha = alpha, 
                    seriestype = seed == :mean ? :line : :scatter,
                    color = color, lc = color, ylim = (0,5.9)
                )
            end
        end
        path = joinpath(base_path, "stepsize-scaling-dim-" * string(dim) * ".pdf")
        savefig(p, path)
    end

    # step size (final tuning round) vs dimension (overlay three models)
    p2 = Plots.plot()
    for model in models 
        for seed in seeds
            df_subset = copy(df)
            if seed != :mean 
                df_subset = filter(:seed => n -> n == seed, df_subset)
            end
            df_subset = filter(:model => n -> n == model, df_subset)
            df_subset = filter(:round => n -> n == maximum(df.round), df_subset)
            seed != :mean ? alpha = 0.25 : alpha = 1.0
            if seed == :mean 
                df_subset = groupby(df_subset, :dim) # group by dim and take average across seeds
                df_subset = combine(df_subset, :step_size => mean, renamecols = false)
            end
            color = findfirst(models .== model)
            Plots.plot!(
                p2, df_subset.dim, df_subset.step_size, 
                xlabel = "Dimension", ylabel = "Step size", 
                label = seed == :mean ? model : "", alpha = alpha, 
                yaxis = :log10, xaxis = :log10,
                seriestype = seed == :mean ? :line : :scatter, color = color,
                lc = color 
            )
            Plots.plot!(
                p2, df_subset.dim, df_subset.dim .^ (-1/3), label = ""
            )
        end
    end
    savefig(p2, joinpath(base_path, "stepsize-scaling.pdf"))
end

function mala_stepsize_performance_plot(df, path) 
    add_mala_to_df!(df) # convert autoMALA to MALA when mala = true
    df = mala_stepsize_performance_clean_df(df)
    sort!(df, :stepsize)

    # compute ratio
    df[!,:leapfrog] = map(zip(df.n_leapfrog,df.ess,df.ess_exact)) do (n,e1,e2)
        n / min(e1,e2) 
    end
    
    samplers = unique(df.sampler)
    seeds = vcat(unique(df.seed), :mean)
    xlabel = "log2(step size / autoMALA step size)"
    for dim in unique(df.dim)
        for model in unique(df.model)
            p = Plots.plot()
            p2 = Plots.plot()
            for seed in seeds
                for sampler in samplers
                    df_subset = copy(df)
                    df_subset = filter(:dim => n -> n == dim, df_subset)
                    if seed != :mean 
                        df_subset = filter(:seed => n -> n == seed, df_subset)
                    end
                    df_subset = filter(:model => n -> n == model, df_subset)
                    seed != :mean ? alpha = 0.25 : alpha = 1.0
                    df_subset = filter(:sampler => n -> n == sampler, df_subset)
                    if seed == :mean 
                        df_subset = groupby(df_subset, :stepsize) # group by step size and take average across seeds
                        df_subset = combine(df_subset, :ess => mean, :leapfrog => mean, :msjd => mean, :acceptance => mean, renamecols = false)
                    end
                    if occursin("autoMALA", sampler) 
                        x = sort(unique(df.stepsize))
                        y = [df_subset.leapfrog[1] for _ in 1:length(x)] 
                        y2 = [df_subset.acceptance[1] for _ in 1:length(x)] 
                    else 
                        x = df_subset.stepsize 
                        y = df_subset.leapfrog
                        y2 = df_subset.acceptance
                    end
                    color = findfirst(samplers .== sampler)
                    Plots.plot!( 
                        p, x, y, 
                        xlabel = xlabel, ylabel = "Leapfrogs per minESS", 
                        yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha,
                        seriestype = seed == :mean ? :line : :scatter,
                        lc = color, color = color 
                    )
                    Plots.plot!( 
                        p2, x, y2, 
                        xlabel = xlabel, ylabel = "MH acceptance probability", 
                        label = seed == :mean ? sampler : "", alpha = alpha,
                        seriestype = seed == :mean ? :line : :scatter,
                        lc = color, color = color 
                    )
                end 
            end
            savefig(p, path * "mala-stepsize-leapfrog-model-" * model * "-dim-" * string(dim) * ".pdf")
            # add optimal acceptance probability for MALA 
            Plots.plot!( 
                p2, sort(unique(df.stepsize)), [0.574 for _i in 1:length(unique(df.stepsize))], 
                label = "", lc = 3
            )
            savefig(p2, path * "mala-stepsize-acceptance-model-" * model * "-dim-" * string(dim) * ".pdf")
        end
    end
end

function mala_stepsize_performance_clean_df(df) 
    seeds = unique(df.seed) 
    models = unique(df.model) 
    dims = unique(df.dim)
    for seed in seeds 
        for model in models 
            for dim in dims 
                df_subset = copy(df)
                df_subset = filter(:mala => n -> n == false, df_subset)
                df_subset = filter(:seed => n -> n == seed, df_subset)
                df_subset = filter(:model => n -> n == model, df_subset)
                df_subset = filter(:dim => n -> n == dim, df_subset)
                @assert nrow(df_subset) == 1 
                base_stepsize = df_subset[1, "stepsize"] 
                for i in 1:nrow(df) # inefficient but it works
                    if (df.seed[i] == seed) && (df.model[i] == model) && (df.dim[i] == dim)
                        df.stepsize[i] = log2(df.stepsize[i] / base_stepsize)
                    end 
                end 
            end
        end 
    end
    return df  
end

function AM_robustness_plot(df,mode="") 
    sort!(df, :step_size)

    # compute ratio
    df[!,:leapfrog] = map(zip(df.n_leapfrog,df.ess,df.ess_exact)) do (n,e1,e2)
        n / min(e1,e2) 
    end

    samplers = unique(df.sampler)
    seeds = vcat(unique(df.seed), :mean)
    base_path = joinpath(
        base_dir(), "deliverables", "AM_robustness" * mode)
    for model in unique(df.model)
        p = Plots.plot()
        p2 = Plots.plot()
        p3 = Plots.plot()
        p4 = Plots.plot()
        p5 = Plots.plot()
        xlabel = "Step size"
        for seed in seeds
            for sampler in samplers
                df_subset = copy(df)
                if seed != :mean 
                    df_subset = filter(:seed => n -> n == seed, df_subset)
                end
                df_subset = filter(:model => n -> n == model, df_subset)
                seed != :mean ? alpha = 0.25 : alpha = 1.0
                df_subset = filter(:sampler => n -> n == sampler, df_subset)
                if seed == :mean 
                    df_subset = groupby(df_subset, :step_size) # group by step size and take average across seeds
                    df_subset = combine(
                        df_subset, :ess => mean, :leapfrog => mean, :msjd => mean, 
                        :acceptance => mean, :final_step_size => mean, renamecols = false)
                end
                color = findfirst(samplers .== sampler)
                Plots.plot!(
                    p, df_subset.step_size, df_subset.ess, 
                    xlabel = xlabel, ylabel = "minESS", 
                    xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha, 
                    lc = color, color = color,
                    seriestype = seed == :mean ? :line : :scatter 
                )
                Plots.plot!( 
                    p2, df_subset.step_size, df_subset.msjd, 
                    xlabel = xlabel, ylabel = "MSJD", 
                    xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha,
                    lc = color, color = color, 
                    seriestype = seed == :mean ? :line : :scatter 
                )
                Plots.plot!( 
                    p3, df_subset.step_size, df_subset.leapfrog, 
                    xlabel = xlabel, ylabel = "Leapfrogs per minESS", 
                    xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha,
                    lc = color, color = color,
                    seriestype = seed == :mean ? :line : :scatter 
                )
                Plots.plot!( 
                    p4, df_subset.step_size, df_subset.acceptance, 
                    xlabel = xlabel, ylabel = "MH acceptance probability", 
                    xaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha,
                    lc = color, color = color,
                    seriestype = seed == :mean ? :line : :scatter 
                )
                if occursin("autoMALA",sampler)
                    Plots.plot!( 
                        p5, df_subset.step_size, df_subset.final_step_size, 
                        xlabel = xlabel, ylabel = "Final step size", 
                        xaxis = :log10, yaxis = :log10, label = seed == :mean ? sampler : "", alpha = alpha,
                        lc = color, color = color,
                        seriestype = seed == :mean ? :line : :scatter,
                        ylims = (minimum(df_subset.step_size), maximum(df_subset.step_size))
                    )
                end
            end 
        end
        savefig(p, joinpath(base_path, "scaling-ess-model-" * model * ".pdf"))
        savefig(p2, joinpath(base_path, "scaling-msjd-model-" * model * ".pdf"))
        savefig(p3, joinpath(base_path, "scaling-leapfrog-model-" * model * ".pdf"))
        savefig(p4, joinpath(base_path, "scaling-acceptance_mean-model-" * model * ".pdf"))
        savefig(p5, joinpath(base_path, "scaling-final_step_size-model-" * model * ".pdf"))
    end
end

function add_mala_to_df!(df::DataFrame) 
    for i in 1:nrow(df) 
        if df[i, "mala"] == true 
            df[i, "sampler"] = "MALA" 
        end
    end
    return df
end

###############################################################################
# sampling utilities
###############################################################################

function model_string(model; dataset=nothing, kwargs...)
    if model == "normal" # dont have the standard normal on Pigeons examples
        return "data {
          int<lower=1> dim;
        }
        parameters {
          vector[dim] x;
        }
        model {
          x ~ std_normal();
        }"
    end
    if startswith(model, "two_component_normal")
        return read(joinpath(
            base_dir(), "stan", "two_component_normal.stan"), String)
    end
    if startswith(model, "horseshoe")
        is_logit = any(Base.Fix1(startswith,dataset), ("prostate", "ionosphere", "sonar"))
        return read(joinpath(
            base_dir(), "stan", "horseshoe_" * (is_logit ? "logit" : "linear") * ".stan"
        ), String)
    end
    if model == "mRNA"
        return read(joinpath(base_dir(), "stan", "mRNA.stan"), String)
    end
    pigeons_stan_dir = joinpath(dirname(dirname(pathof(Pigeons))),"examples","stan")
    if startswith(model, "eight_schools_") 
        return read(joinpath(pigeons_stan_dir,"$model.stan"), String)
    end
    model_class = first(split(model,"_"))
    if model_class in ("banana","funnel") 
        return read(joinpath(pigeons_stan_dir,"$model_class.stan"), String)
    end
    error("model_string: model $model unknown")
end

function stan_data(model::String; dataset=nothing, dim=nothing, scale=nothing) 
    if model in ("funnel", "banana")
        Dict("dim" => dim-1, "scale" => scale)
    elseif model in ("funnel_scale", "banana_scale") 
        Dict("dim" => 1, "scale" => inv(dim))
    elseif model == "normal"
        Dict("dim" => dim) 
    elseif model == "two_component_normal_scale"
        s_lo, s_hi = two_component_normal_stdevs(dim)
        Dict("n" => 1, "s_lo" => s_lo, "s_hi" => s_hi)
    elseif model == "horseshoe"
        x,y = isnothing(dim) ? make_HSP_data(dataset) : make_HSP_data(dataset,dim) # interpret dim as n_obs
        Dict("n" => length(y), "d" => size(x,2), "x" => x, "y" => y)
    elseif startswith(model,"eight_schools")
        Dict("J" => 8, "y" => [28, 8, -3, 7, -1, 1, 18, 12],
        "sigma" => [15, 10, 16, 11, 9, 11, 10, 18])
    elseif model == "mRNA"
        dta = DataFrame(CSV.File(joinpath(base_dir(), "data", "transfection.csv")))
        Dict("N" => nrow(dta), "ts" => dta[:,1], "ys" => dta[:,3])
    else
        error("stan_data: unknown model $model") 
    end 
end 

function AM_normal_574_plot(df,mode="") 
    sort!(df, :dim) 
    seeds = vcat(unique(df.seed), :mean)
    p = Plots.plot()
    for seed in seeds
        df_subset = copy(df)
        if seed != :mean 
            df_subset = filter(:seed => n -> n == seed, df_subset)
        end
        seed != :mean ? alpha = 0.25 : alpha = 1.0
        if seed == :mean 
            df_subset = groupby(df_subset, :dim) # group by dim and take average across seeds
            df_subset = combine(df_subset, :acceptance => mean, renamecols = false)
        end
        Plots.plot!( 
            p, df_subset.dim, df_subset.acceptance, 
            xlabel = "Dimension", ylabel = "MH acceptance probability", 
            xaxis = :log10, alpha = alpha, lc = 1, label = (seed == :mean) ? "autoMALA" : "", 
            color = 1, seriestype = seed == :mean ? :line : :scatter,
            ylims = (0, 1)
        )
    end
    Plots.hline!(p, [0.574], lc = 3, label = "Theoretical optimal value") # optimal MALA acceptance probability 
    path = joinpath(
        base_dir(), "deliverables", "AM_normal_574" * mode, "AM-normal-574-acceptance.pdf") 
    savefig(p, path)
end

###############################################################################
# sampling from models for real data
###############################################################################

# build the horseshoe prior target with varying number of observations
load_HSP_df(dataset::String) = 
    DataFrame(CSV.File(
        joinpath(base_dir(), "data", dataset * ".csv") ))
make_HSP_data(dataset::String, n_obs::Int=typemax(Int)) = 
    make_HSP_data(dataset,load_HSP_df(dataset),n_obs)
function make_HSP_data(dataset::String, df::DataFrame, n_obs::Int)
    iszero(n_obs) && return (zeros( ( n_obs,size(df,2)-1 ) ), Int64[])
    n = min(n_obs, size(df, 1))
    if startswith(dataset,"prostate")
        x = Matrix(df[1:n,2:end])
        y = df[1:n,1]
    elseif startswith(dataset,"ionosphere")
        x = Matrix(hcat(df[1:n,1], df[1:n,3:(end-1)])) # col 2 is constant
        y = Int.(df[1:n,end] .== "g")
    elseif startswith(dataset,"sonar")
        x = Matrix(df[1:n,1:(end-1)])
        y = Int.(df[1:n,end] .== "Mine")
    end
    x,y
end
function make_HSP_target(dataset::String, n_obs::Int=typemax(Int))
    xmat,y = make_HSP_data(dataset, n_obs)
    d = size(xmat,2)
    json_str = if iszero(n_obs)
        Pigeons.json(; n=n_obs,d=d,x="[[]]",y="[]")
    else
        x = [copy(r) for r in eachrow(xmat)]
        Pigeons.json(; n=length(y), d=d, x=x, y=y)
    end
    is_logit = any(Base.Fix1(startswith,dataset), ("prostate", "ionosphere", "sonar"))
    StanLogPotential(joinpath(
        base_dir(), "stan", "horseshoe_" * (is_logit ? "logit" : "linear") * ".stan"
    ), json_str)
end

# Boxplots
const DEFAULT_COLORBLIND_PALETTE = [colorant"#785EF0", colorant"#DC267F", colorant"#FE6100", colorant"#FFB000"] # colorant"#648FFF" 
dataset_nickname(d::AbstractString) = d=="sonar" ? "Sonar" : (d=="prostate_small" ? "Prostate" : "Ion.")
function make_boxplots(df::DataFrame; fn_end = ".pdf")
    only_two_samplers = length(unique(df.sampler)) == 2
    colors = only_two_samplers ? :auto : DEFAULT_COLORBLIND_PALETTE
    if !only_two_samplers # use better descriptors for various automalas
        map!(df[!,:sampler],df[!,:sampler]) do s
            s == "autoMALA" ? "AM smooth" : (s == "autoMALA_fixed" ? "AM single" : (s == "mix_autoMALA" ? "AM mixture" : s))
        end
    end

    # preprocessing
    sort!(df)
    model = first(df.model)
    is_funnel = occursin("funnel",model)
    is_banana= occursin("banana",model)
    is_banana && (df.dim .= log2.(df.dim)) # boxplot not working with xaxis=:log 
    is_hsp = occursin("horseshoe",model) && hasproperty(df, :dataset)
    is_2_comp_norm = occursin("two_component_normal",model)
    xvar = :dim
    if is_hsp
        xvar = :data_nobs
        df[!,xvar] = map(
            t -> (dataset_nickname(t[1]) * "\n" * (t[2] > 100 ? "full" : string(t[2]))),
            zip(df.dataset,df.dim)
        )
        sort!(df, xvar)
    end
    df[!,:min_ess] = min.(df.ess, df.ess_exact)
    df[!,:nleap_to_min_ess] = df.n_leapfrog ./ df.min_ess
    df_means = combine(
        groupby(df, [xvar, :sampler]),
        :margin1_mean => mean,
        :margin1_var => mean,
        :n_leapfrog => mean,
        :min_ess => mean,
        :nleap_to_min_ess => mean, 
        renamecols=false)
    sort!(df_means)

    # common properties
    size   = (650,300)
    xlab   = is_hsp ? "Dataset" : "Inverse scale" * (is_banana ? " (log₂)" : "")
    mar    = 15px
    path   = joinpath(base_dir(), "deliverables", "AM_" * model)
    n_samplers = length(unique(df.sampler))

    # plots for known margins
    if !is_hsp
        # margin1 mean
        margin_idx = is_2_comp_norm ? 2 : 1
        p=@df df groupedboxplot(
            :dim, 
            :margin1_mean, 
            group=:sampler,
            bar_position = :dodge,
            size=size,
            xlab=xlab,
            palette=colors,
            ylab="Margin $margin_idx mean",
            left_margin = mar, bottom_margin = mar,
        )
        if !only_two_samplers
            @df df_means plot!(p,
                :dim,
                :margin1_mean,
                group=:sampler,
                palette=colors,
                linewidth=2,
                label=""
            )
        end
        savefig(p,joinpath(path, "boxplots-margin-mean" * fn_end))

        # margin1 var
        p=@df df groupedboxplot(
            :dim, 
            :margin1_var, 
            group=:sampler,
            bar_position = :dodge, 
            size=size,
            xlab=xlab,
            legend = is_2_comp_norm ? :topleft : :best,
            yaxis= is_2_comp_norm ? :log : :identity,
            palette=colors,
            ylab="Margin $margin_idx var",
            left_margin = mar, bottom_margin = mar,
        )
        if !only_two_samplers
            @df df_means plot!(p,
                :dim,
                :margin1_var,
                group=:sampler,
                yaxis= is_2_comp_norm ? :log : :identity,
                palette=colors,
                linewidth=2,
                label=""
            )
        end
        if is_2_comp_norm
            dim_vals = sort(unique(df.dim))
            plot!(dim_vals, 10. .^ (2*dim_vals), label = "true",
            linestyle=:dash, color=colorant"#648FFF")
        end
        savefig(p,joinpath(path, "boxplots-margin-var" * fn_end))
    end

    # n_leapfrog
    p=@df df groupedboxplot(
        (is_hsp ? :data_nobs : :dim), 
        :n_leapfrog, 
        group=:sampler,
        bar_position = :dodge, 
        legend = is_hsp || is_2_comp_norm ? :outerright : :best,
        yaxis= :log,
        palette=colors,
        size=size,
        xlab=xlab,
        ylab="Total number of leapfrog steps",
        left_margin = mar, bottom_margin = mar,
    )
    if !only_two_samplers && is_hsp
        plot!(p,
            repeat(first(first(xticks(p))),inner=n_samplers),
            df_means.n_leapfrog,
            group=df_means.sampler,
            yaxis=:log,
            palette=colors,
            linewidth=2,
            label=""
        )
    elseif !only_two_samplers
        @df df_means plot!(p,
            :dim,
            :n_leapfrog,
            group=:sampler,
            yaxis=:log,
            palette=colors,
            linewidth=2,
            label=""
        )
    end
    savefig(p,joinpath(path, "boxplots-n_leapfrog" * fn_end))

    # min ess
    p=@df df groupedboxplot(
        (is_hsp ? :data_nobs : :dim), 
        :min_ess,
        group=:sampler,
        bar_position = :dodge, 
        legend = is_2_comp_norm ? :outerright : :best,
        yaxis= :log,
        palette=colors,
        size=size,
        xlab=xlab,
        ylab="minESS",
        left_margin = mar, bottom_margin = mar
    )
    if !only_two_samplers && is_hsp
        plot!(p,
            repeat(first(first(xticks(p))),inner=n_samplers),
            df_means.min_ess,
            group=df_means.sampler,
            yaxis=:log,
            palette=colors,
            linewidth=2,
            label=""
        )
    elseif !only_two_samplers
        @df df_means plot!(p,
            :dim,
            :min_ess,
            group=:sampler,
            yaxis=:log,
            palette=colors,
            linewidth=2,
            label=""
        )
    end
    savefig(p,joinpath(path, "boxplots-miness" * fn_end))

    # nleap to miness
    p=@df df groupedboxplot(
        (is_hsp ? :data_nobs : :dim), 
        :nleap_to_min_ess, 
        group=:sampler,
        bar_position = :dodge, 
        legend = (only_two_samplers || is_funnel) ? :bottomright : (is_2_comp_norm ? :topleft : (is_hsp ? :outerright : :best)),
        yaxis= :log,
        palette=colors,
        size=size,
        xlab=xlab,
        ylab="Leapfrogs per minESS",
        left_margin = mar, bottom_margin = mar,
    )
    if !only_two_samplers && is_hsp
        plot!(p,
            repeat(first(first(xticks(p))),inner=n_samplers),
            df_means.nleap_to_min_ess,
            group=df_means.sampler,
            yaxis=:log,
            palette=colors,
            linewidth=2,
            label=""
        )
    elseif !only_two_samplers
        @df df_means plot!(p,
            :dim,
            :nleap_to_min_ess,
            group=:sampler,
            yaxis=:log,
            palette=colors,
            linewidth=2,
            label=""
        )
    end
    savefig(p,joinpath(path, "boxplots-nleap_to_min_ess" * fn_end))
end

# Two component normal for testing preconditioner
function make_2_comp_norm_target(n, exponent)
    s_lo, s_hi = two_component_normal_stdevs(exponent)
    json_str = Pigeons.json(; n=n, s_lo=s_lo, s_hi=s_hi)
    StanLogPotential(joinpath(
        base_dir(), "stan", "two_component_normal.stan"
    ), json_str)
end

# mRNA Transfection
# missing functions in Pigeons for DistributionLogPotentials
function sample_iid!(ref::DistributionLogPotential, replica::Pigeons.Replica{<:Pigeons.StanState}, shared)
    rand!(replica.rng, ref.dist, replica.state.unconstrained_parameters)
end
(ref::DistributionLogPotential)(x::Pigeons.StanState) = logpdf(ref.dist, x.unconstrained_parameters)

# custom initialization for mRNA Transfection example
const TPriorMRNA = DistributionLogPotential{Distributions.ProductDistribution{1, 0, NTuple{5, Uniform{Float64}}, Continuous, Float64}}
function initialization(::Inputs{T,V,E,TPriorMRNA}, ::SplittableRandom, ::Int64) where {T, V, E}
    Pigeons.StanState([0.302164,  0.705052,  -0.114732,   -0.661198,  -0.94706])
end
