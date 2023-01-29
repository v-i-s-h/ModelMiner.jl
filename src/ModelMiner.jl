module ModelMiner
import MLJ
import MLJModels
import DataFrames: DataFrame


"""
    _load_mlj_packages(algos::Vector{NamedTuple})

Load avaliable packages from a NamedTuple of algorithms.
`algos` can be the results of `MLJ.models()`. The only requirement
in `algos` is that each entry should have `name` and `package_name`.

Note: This function will try to load all given algorithms if it is available
on the packages environment.
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
        catch e
            if !isa(e, ArgumentError)
                # If the error is not related to importing the MLJ Modules
                rethrow()
            end
        end
    end

    return loaded_models
end

# Load all available packages
const LOADED_MODELS = _load_mlj_packages(MLJ.models())

const BLACKLIST_MODELS = [
    (name="MultitargetNeuralNetwork", package_name="BetaML")
]


"""
    mine()

Train all avaliable models from the environment.
"""
function _mine(data...; measures=[])
    match_fn = MLJ.matching(data...)
    available_algos = MLJ.models(
        m -> match_fn(m) 
        && (MLJModels.load_path(m) ∈ LOADED_MODELS)
        && (m.name ∉ [bl.name for bl in BLACKLIST_MODELS])
    )


    results = [] # To hold the results from each model training

    for _a ∈ available_algos
        _model = MLJModels.load_path(_a)
        model = _model |> Meta.parse |> eval
        machine = model()

        try
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
        catch
            # Some error happened in training this model.
            # Let's skip this
            # @warn "Skipping $(_a.name) ($(_a.package_name)) due to internal errors"
        end
    end

    return results
end

"""
    mine(X, y::Vector{T}) <: AbstractFloat

Train regression models (where target `y` is a real value).
"""
function mine(X, y::Vector{T}) where {T<:AbstractFloat}
    return _mine(X, y; measures=[MLJ.rms]) |> DataFrame
end

"""
    mine(X, y)

Train non-regression (classification) models (where `y` is not a floating point
variable).
"""
function mine(data...)
    return _mine(data...; measures=[MLJ.accuracy, MLJ.multiclass_f1score]) |> DataFrame
end

export mine
end
