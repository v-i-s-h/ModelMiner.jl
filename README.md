# ModelMiner.jl 
_One package to train them all._

ModelMiner.jl aims to provide an easy use interface to train multiple machine learning
models with a single function all. It provides a `mine(...)` interface through
which users can train multiple classification and regression models easily. The goal
is to quickly verify which models performs better on data without much effort.


## Usage
### Classification example
1. Prepare your data
```julia
using MLJ, RDatasets
data = dataset("datasets", "iris")
y, X = MLJ.unpack(data, ==(:Species))
```
2. Call `mine(...)` to train and evaluate models
```julia
using ModelMiner
results = mine(X, y)
```
The output will a dataframe with performance measures according to the data
```
7×3 DataFrame
 Row │ name                             Accuracy  MulticlassFScore 
     │ String                           Float64   Float64          
─────┼─────────────────────────────────────────────────────────────
   1 │ AdaBoostStumpClassifier          0.94              0.927209
   2 │ ConstantClassifier               0.266667          0.140233
   3 │ DecisionTreeClassifier           0.966667          0.965303
   4 │ DecisionTreeClassifier           0.92              0.918642
   5 │ DeterministicConstantClassifier  0.266667          0.140063
   6 │ KernelPerceptron                 0.94              0.937882
   7 │ LinearPerceptron                 0.813333          0.771626
   8 │ LinearSVC                        0.96              0.963594
   9 │ LogisticClassifier               0.973333          0.968547
  10 │ MultinomialClassifier            0.953333          0.952808
  11 │ NeuralNetworkClassifier          0.533333          0.405575
  12 │ NuSVC                            0.966667          0.96627
  13 │ Pegasos                          0.6               0.484509
  14 │ RandomForestClassifier           0.953333          0.957148
  15 │ RandomForestClassifier           0.953333          0.948225
  16 │ SVC                              0.96              0.957057
  17 │ XGBoostClassifier                0.94              0.939973
```

### Regression example
1. Prepare your data
```julia
data = dataset("MASS", "Boston")
y, X = MLJ.unpack(data, ==(:MedV))
```
2. Train models and evaluate models
```julia
using ModelMiner
results = mine(X, y)
```

Results:
```
8×2 DataFrame
 Row │ name                            RootMeanSquaredError 
     │ String                          Float64              
─────┼──────────────────────────────────────────────────────
   1 │ ConstantRegressor                            9.22236
   2 │ DecisionTreeRegressor                        5.0619
   3 │ DecisionTreeRegressor                        4.51225
   4 │ DeterministicConstantRegressor               9.19367
   5 │ GaussianMixtureRegressor                     8.18446
   6 │ NeuralNetworkRegressor                      20.2011
   7 │ RandomForestRegressor                        3.98226
   8 │ RandomForestRegressor                        3.38003
```

## Acknowledgements
Thanks to the developers of [LazyPredict](https://github.com/shankarpandala/lazypredict)
for inspiration and [@ablaom](https://github.com/ablaom) for helping me overcome
[some challenges](https://discourse.julialang.org/t/using-load-from-mlj-inside-a-package/93413) in developing.