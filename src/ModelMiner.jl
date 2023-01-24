module ModelMiner
import MLJ
import MLJModels


"""
    _load_mlj_packages(algos::Vector{NamedTuple})

Load avaliable packages from a NamedTuple of algorithms.
`algos` can be the results of `MLJ.models()`. The only requirement
in `algos` is that each entry should have `name` and `package_name`.

Note: This function will try to load all given algorithms if it is available
on the current environment. The user is required to keep the environment updated
with the packages they want to train the models with.
"""
function _load_mlj_packages(algos)
    # Find all MLJPackages
    loaded_models = []
    for a ∈ algos
        # path = MLJModels.load_path(a.name; pkg=a.package_name)
        algo_path = MLJModels.load_path(a)

        # Package name
        pkg_name = split(algo_path, ".") |> first

        try
            eval(:(import $(Symbol(pkg_name))))
            push!(loaded_models, algo_path)
            # @info "Loaded " * algo_path
        catch e
            if isa(e, ArgumentError)
                # @warn "Skipping " * algo_path
            else
                rethrow()
            end
        end
    end

    return loaded_models
end

# Load all available packages
const LOADED_MODELS = _load_mlj_packages(MLJ.models())


"""
    mine()

Train all avaliable models from the environment.
"""
function _mine(data...; measures=[])
    match_fn = MLJ.matching(data...)
    available_algos = MLJ.models(
        m -> match_fn(m) && (MLJModels.load_path(m) ∈ LOADED_MODELS)
    )


    results = [] # To hold the results from each model training

    for _a ∈ available_algos
        _model = MLJModels.load_path(_a)
        # @info "Evaluating " * _model
        model = _model |> Meta.parse |> eval
        machine = model()

        # Train and evaluate
        r = MLJ.evaluate(
            machine,
            data...;
            resampling=MLJ.CV(shuffle=true),
            verbosity=0,
            measures=measures
        )

        # Save results
        push!(
            results, (
                name=_a.name,
                NamedTuple(zip(
                    measures .|> typeof .|> t -> t.name.name, # get measurement name as symbol
                    r.measurement # corresponding measurement value
                ))...
            )
        )
    end

    return results
end

"""
    mine(X, y::Vector{T}) <: AbstractFloat

Train regression models (where target `y` is a real value).
"""
function mine(X, y::Vector{T}) where {T<:AbstractFloat}
    return _mine(X, y; measures=[MLJ.rms])
end

"""
    mine(X, y)

Train classification models.
(because regression models are handled above?)
"""
function mine(data...)
    return _mine(data...; measures=[MLJ.accuracy, MLJ.multiclass_f1score])
end

export mine
end
