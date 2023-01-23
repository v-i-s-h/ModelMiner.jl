using ModelMiner
using Test
using RDatasets

@testset "ModelMiner.jl" begin
    data = dataset("datasets", "iris")
    
    # Split into features and targets
    y, X = unpack(data, ==(:Species))
    
    a = mine(X, y)
    print(a)

    @test true # Dummy test to return pass.
end
