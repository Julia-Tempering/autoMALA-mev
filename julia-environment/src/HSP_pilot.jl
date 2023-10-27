###############################################################################
# Logistic regression with horseshoe prior
# Preparation of datasets
###############################################################################

include("AM_scaling_utils.jl") # packages loaded inside this file

#=
prostate cancer data in https://github.com/charlesm93/laplace_manuscript
we subset the predictors to make it manageable
=#
# find relevant stuff with Lasso
using Lasso, GLM, SparseArrays
cancer_df = DataFrame(CSV.File(joinpath(base_dir(),"data","prostate.csv")))
y = cancer_df[:,1]
x = Matrix(cancer_df[:,2:end])
lasso_fit = fit(LassoPath, x, y, Binomial(), LogitLink())
C = lasso_fit.coefs
rows = rowvals(C)
nz_idx_vecs = [rows[nzrange(C, j)] for j in 1:last(size(C))]
best_pred_subset = union(nz_idx_vecs...) # union across all subsets in the path
@assert 1816 in best_pred_subset && 2586 in best_pred_subset # (1816, 2586) marked as important in https://github.com/charlesm93/laplace_manuscript/blob/b51a7ade9f28caf0cd722ed1f052b79b2d7ca107/format_data.R#L56

# add some garbage variables
Random.seed!(3)
target_n_pred = 50
n_bad = target_n_pred - length(best_pred_subset)
bad_pred_set = rand(setdiff(1:size(x)[2], best_pred_subset), n_bad)
pred_set = sort(union(best_pred_subset, bad_pred_set))
@assert length(pred_set) == target_n_pred

# save
x_small = x[:,pred_set]
df_small = DataFrame()
df_small.y=y
df_small = hcat(df_small, DataFrame(x_small,"x" .* string.(pred_set)) )
rand_idx = randperm(size(df_small)[1]) # randomize the rows in order to be able to take unbiased subsets via deterministic ranges 1:n
df_small = df_small[rand_idx,:]
CSV.write(joinpath(base_dir(),"data","prostate_y_x_small.csv"), df_small)

#=
sonar dataset
=#
# randomize the rows in order to be able to take unbiased subsets via deterministic ranges 1:n
Random.seed!(3)
sonar_df = DataFrame(CSV.File(joinpath(base_dir(),"data","sonar.csv"),header=false))
rand_idx = randperm(size(sonar_df,1))
sonar_df = sonar_df[rand_idx,:]
CSV.write(joinpath(base_dir(),"data","sonar.csv"), sonar_df)

#=
ionosphere dataset
=#
# randomize the rows in order to be able to take unbiased subsets via deterministic ranges 1:n
Random.seed!(3)
iono_df = DataFrame(CSV.File(joinpath(base_dir(),"data","ionosphere.csv"),header=false))
rand_idx = randperm(size(iono_df,1))
iono_df = iono_df[rand_idx,:]
CSV.write(joinpath(base_dir(),"data","ionosphere.csv"), iono_df)