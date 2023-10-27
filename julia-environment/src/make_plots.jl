include("AM_scaling_utils.jl") # packages loaded inside this file

# 2 samplers plots
AM_version = "mix_autoMALA" # choose the version of autoMALA to show

for model in ("AM_funnel_scale", "AM_banana_scale")
    make_boxplots(filter_single_AM_version!(get_summary_df(model), AM_version))
end

model = "AM_horseshoe"
make_boxplots(filter_single_AM_version!(get_summary_df(model), AM_version))

model = "mala_stepsize_performance"
path = joinpath(base_dir(), "deliverables", model * "/")
mala_stepsize_performance_plot(filter_single_AM_version!(get_summary_df(model), AM_version), path)

model = "AM_stepsize_scaling"
AM_stepsize_scaling_plot(filter_single_AM_version!(get_summary_df(model), AM_version))

model = "AM_normal_574"
AM_normal_574_plot(filter_single_AM_version!(get_summary_df(model), AM_version))

model = "AM_robustness"
AM_robustness_plot(filter_single_AM_version!(get_summary_df(model), AM_version))

for model in ("AM_funnel_scale", "AM_banana_scale","AM_funnel_highdim","AM_banana_highdim","AM_normal_highdim")
    df = filter_single_AM_version!(get_summary_df(model), AM_version)
    path = joinpath(base_dir(), "deliverables", model * "/")
    AM_scaling_plot(df, path)
end

# many samplers plots
for model in ("AM_funnel_scale", "AM_two_component_normal_scale")
    make_boxplots(get_summary_df(model),fn_end="-all_automalas.pdf")
end
