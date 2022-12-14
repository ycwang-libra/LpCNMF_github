# LpCNMF
This is an implementation for our paper "Constrained Nonnegative Matrix Factorization based on Label Propagation for Data Representation".

## Datasets
Collected [datasets](datasets) can be loaded in Matlab.

## Code
Add the [algorithms](code/baselines/) and [tools](code/dependencies/) into the path of Matlab and use the code in [demo](code/demo/) to accomplish relevant experiments.

## Tested Results
You can also use the test results directly in file [Saved Results](Saved_Results) which includes:
* [comparison results](Saved_Results/compare_results/) with all state-of-the-art methods and [ablation label propagation](Saved_Results/ablation_results_label_propagation/) with ablate the label propagation (Data format: `number of cluster (9) * number of test(10)`)
* hyperparameters selection with [label proportion](Saved_Results/labelproportion_results/), [regularization](Saved_Results/regularization_results/) and [heat kernel](Saved_Results/heatkernel_results/) (Data format: `number of cluster (9) *  number of parameters(7 or 9) * number of test(10)`)
* relationships of label proportion with two hyperparameters [label proportion with heat kernel](Saved_Results/labelproportion_with_heat_kernel/) and [label proportion with regularization](Saved_Results/labelproportion_with_regularization/) (Data format: `number of cluster (9) * number of label percent(9) * number of hyperparameters(7) * number of test(5)`)
* [time consumption result ](Saved_Results/time_consume_results/)(Data format: `1 * number of trails(10)`)