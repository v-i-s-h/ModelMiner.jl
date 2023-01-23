module ModelMiner
# using MLJ
import MLJ: unpack, models, @load, load, matching
import MLJ: accuracy, multiclass_f1score

blacklist_package = ["MLJFlux"]

function mine(X, y)
    match_fn = matching(X, y)
    available_algos = models(
        m -> match_fn(m) && m.is_pure_julia && !(m.package_name ∈ blacklist_package)
    )

    measures = [accuracy, multiclass_f1score]
    results = []
    for algo_info ∈ available_algos
        @info "Training " * algo_info.name * " -- " * algo_info.package_name

        # Make model
        algo = load(algo_info.name; pkg=algo_info.package_name)
        machine = algo()

        # Train and evaluate
        r = evaluate(
            machine,
            X, y,
            resampling=CV(shuffle=true),
            measures=measures,
            verbosity=verbosity
        )

        # Save results
        push!(
            results, (
                name=algo_info.name,
                NamedTuple(zip(
                    measures .|> typeof .|> t -> t.name.name, # get measurement name as symbol
                    r.measurement # corresponding measurement value
                ))...
            )
        )
    end

    return results
end

export mine, unpack
end
