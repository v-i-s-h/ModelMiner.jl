module ModelMiner

import Pkg
import MLJ
import MLJModels
import DataFrames: DataFrame

LOADED_MODULES = []

"""
Source: Pkg package (modified)
"""
function installed()
    deps = Pkg.dependencies()
    installed_pkgs = Vector{String}()
    for (uuid, dep) in deps
        dep.is_direct_dep || continue
        dep.version === nothing && continue
        # installs[dep.name] = dep.version::VersionNumber
        push!(installed_pkgs, dep.name)
    end
    return installed_pkgs
end

function __init__()
    installed_pkgs = installed()

    algos = MLJ.models()

    for a ∈ algos
        algo_path = MLJModels.load_path(a)

        # Extract package name
        pkg_name = split(algo_path, ".") |> first

        # In the package is available in current environment, load it
        if pkg_name ∈ installed_pkgs
            pkg_name ∉ LOADED_MODULES && begin
                @eval Main begin
                    import $(Symbol(pkg_name))
                end

                push!(LOADED_MODULES, pkg_name)
            end
        end
    end
end


"""
    mine()

Train all avaliable models from the environment.
"""
function _mine(data...; measures=[])
    match_fn = MLJ.matching(data...)
    available_algos = MLJ.models(
        m -> match_fn(m) 
        && (
            split(MLJModels.load_path(m), ".")[1] ∈ LOADED_MODULES
        )
    )

    results = [] # To hold the results from each model training

    for _a ∈ available_algos
        _model = "Main." * MLJModels.load_path(_a)
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
            @warn "Skipping $(_a.name) ($(_a.package_name)) due to internal errors"
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
