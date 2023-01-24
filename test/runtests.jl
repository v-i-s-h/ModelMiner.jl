using ModelMiner
using Test
import RDatasets: dataset
import MLJ


function is_models_in_results(results, expected_models)
    return [r.name for r ∈ results] |> # Collect all model names in results
           models -> [ # Check is each of expected is in results
                expected ∈ models for expected ∈ expected_models
           ] |>
           r -> reduce(&, r) # Combine to all checks
end

@testset "ModelMiner.jl" begin
    @testset "Classifier test" begin
        data = dataset("datasets", "iris")

        # Split into features and targets
        y, X = MLJ.unpack(data, ==(:Species))

        # For test, we will check if it has results for all the models
        # from Test environment
        expected_models = [
            "ConstantClassifier",
            "DeterministicConstantClassifier",
            # "LogisticClassifier",
            # "MultinomialClassifier",
            # "AdaBoostStumpClassifier", 
            # "DecisionTreeClassifier",
            # "RandomForestClassifier"
        ]

        # Train models
        results = mine(X, y)

        # Check if all expected models are in the results
        @test is_models_in_results(results, expected_models)

        # `results` is a NamedTuple with name, accuracy and F1 score
        # TODO: Check if results have `Accuracy` and `MulticlassFScore`
    end

    @testset "Regressor test" begin
        data = dataset("datasets", "anscombe")

        # Split into features and targets
        y, X = MLJ.unpack(data, ==(:Y4))

        expected_models = [
            "ConstantRegressor",
            "DeterministicConstantRegressor",
            # "DecisionTreeRegressor",
            # "RandomForestRegressor"
        ]

        # Train models
        results = mine(X, y)

        @test is_models_in_results(results, expected_models)

        # TODO: Check if result has `RootMeanSquareError`
    end
end
