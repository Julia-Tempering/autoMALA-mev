# using MCMCChains
# using MCMCDiagnosticTools

# bias = 0.5 # ESS should drop as bias increases
# sample = [[randn() + bias] for _ in 1:1_000]
# ess_df = MCMCChains.ess(Chains(sample); kind=:basic, exact_means = [0.0]) # basic is usual ESS. default is :bulk which applies ESS to rank-normalized samples
# result = ess_df.nt.ess