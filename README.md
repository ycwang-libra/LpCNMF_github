# LpCNMF
This is an implementation for our paper "Constrained Nonnegative Matrix Factorization based on Label Propagation for Data Representation".

## Datasets
Collected [datasets](datasets) can be loaded in Matlab.

## Code
Add the [algorithms](code/baselines/) and [tools](code/dependencies/) into the path of Matlab and use the code in [demo](code/demo/) to accomplish relevant experiments.

## Tested Results
You can use the test results directly in file [Saved Results](Saved_Results) which include:
* [comparison results](Saved_Results/compare_results/) with all state-of-the-art methods
* hyperparameters selection with [label proportion](Saved_Results/labelproportion_results/), [regularization](Saved_Results/regularization_results/) and [heat kernel](Saved_Results/heatkernel_results/)
* relationships of hyperparameters [label proportion with heat kernel](Saved_Results/labelproportion_with_heat_kernel/) and [label proportion with regularization](Saved_Results/labelproportion_with_regularization/)