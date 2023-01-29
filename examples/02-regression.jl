using MLJ
using RDatasets
using ModelMiner

# Regression
data = dataset("MASS", "Boston")
y, X = MLJ.unpack(data, ==(:MedV))

results = mine(X, y)

