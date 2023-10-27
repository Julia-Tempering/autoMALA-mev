using Pigeons 

dim = 2^10 
n_rounds = 15
pt = pigeons(
    target = Pigeons.ScaledPrecisionNormalPath(1.0, 1.0, dim), 
    n_rounds = n_rounds,
    n_chains = 1, 
    explorer = AutoMALA(),
    record = [record_default(); Pigeons.explorer_acceptance_pr]
)

println(Pigeons.explorer_mh_prs(pt)[1])


# dim       n_rounds
# 2^10      11
# 2^15 