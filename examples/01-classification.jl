using MLJ
using RDatasets
using ModelMiner

# Classification
data = dataset("datasets", "iris")
y, X = MLJ.unpack(data, ==(:Species))

results = mine(X, y)

