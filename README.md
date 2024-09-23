
# Short-term Memory Active Learning (SMAL)
As active learning campaigns progress, it has been observed that performance can have the potential to decrease after a "turning point" of maximum performance (Wen et al.). In standard active learning, data is continuously added in a linear fashion. It is hypothesized that this could be problematic since selection functions are likely imperfect, especially during early stages of active learning campaigns where data is limited leading to a poorer understanding of a particular domain. SMAL was developed to augment standard active learning approaches by implementing backward forgetting of training data based on various measures of sample utility. Forgetting data leads to restricted training set sizes making models more compact and less biased, while leading to equivalent or improved overall performance. Additionally, the re-integration of prior experimental data reduces labeling costs, and enhances training set diversity and quality.
* insert figure 1 *

This package was built on [MolALKit](https://github.com/RekerLab/MolALKit).

## Installation
```commandline
pip install numpy==1.22.3 git+https://gitlab.com/Xiangyan93/graphdot.git@feature/xy git+https://github.com/bp-kelley/descriptastorus git+https://github.com/Xiangyan93/chemprop.git@molalkit
pip install mgktools
```

## Capabilities and Utility
### Forgetting Protocols for SMAL:
- 'forget_random': Randomly forget selected training datapoints
- 'forget_first': Forget the training datapoint that has been in the training dataset the longest (first-in-first-out)
- 'max_oob_uncertainty': Forget the training datapoint with the most uncertain prediction (out-of-bag)
- 'min_oob_uncertainty': Forget the training datapoint with the least uncertain prediction
- 'max_oob_uncertainty_correct' and 'max_oob_uncertainty_incorrect': Forget the training datapoint with the most uncertain prediction, considering if its class label was also predicted correctly/incorrectly
- 'min_oob_uncertainty_correct' and 'min_oob_uncertainty_incorrect': Forget the training datapoint with the least uncertain prediction, considering if its class label was also predicted correctly/incorrectly
* insert graphical abstract *

### When to Forget:
- 'forget_size': In this study, this was implemented as the training set size associated with the "turning point" of observed standard active learning trajectories on the same dataset. Once this training set size is reached, data will be simultaneously added and forgotten at each iteration.
- 'forget_ratio': Start forgetting when the training set size reaches some percentage of total data from the original pool set.

### Perturbing Data:
- 'error_rate': Fraction of training data to perturb within dataset.

### Example
Experiment: Applying SMAL to the HIV dataset with the Minimum Out-of-Bag Uncertainty forgetting protocol

Additional parameters: Random forest model with Morgan fingerprint descriptors, explorative active learning protocol, 20% introduced error, 50:50 scaffold random train:test split, 1554 forget size (determined prior)

```commandline
python3 SMAL.py --data_public hiv --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride 1 --seed 0 --save_dir experiment_example --n_jobs 1 --forget_size 1554 --forget_protocol min_oob_uncertainty --error_rate 0.2
```

## Active Learning and Other Usage
More information can be found at [MolALKit](https://github.com/RekerLab/MolALKit).

## References
Wen, Y., Li, Z., Xiang, Y., & Reker, D. (2023). Improving Molecular Machine Learning Through Adaptive Subsampling with Active Learning.

