using Pigeons 

dim = 2 
n_rounds = 18
step_sizes = 2.0 .^(-10:1) 

for step_size in step_sizes
    # autoMALA
    pt = pigeons(
        target = ScaledPrecisionNormalPath(1.0, 1.0, dim), 
        n_rounds = n_rounds,
        n_chains = 1, 
        explorer = AutoMALA(step_size = step_size),
        record = [record_default(); Pigeons.explorer_acceptance_pr], 
        show_report = false
    )
    println("autoMALA")
    println("")
    println("Acceptance probability:")
    println(Pigeons.explorer_mh_prs(pt)[1])
    println("Step size:")
    println(pt.shared.explorer.step_size)

    # MALA 
    pt = pigeons(
        target = ScaledPrecisionNormalPath(1.0, 1.0, dim), 
        n_rounds = n_rounds,
        n_chains = 1, 
        explorer = MALA(step_size = step_size),
        record = [record_default(); Pigeons.explorer_acceptance_pr], 
        show_report = false
    )
    println("MALA")
    println("")
    println("Acceptance probability:")
    println(Pigeons.explorer_mh_prs(pt)[1])
    println("Step size:")
    println(pt.shared.explorer.step_size)
end 



