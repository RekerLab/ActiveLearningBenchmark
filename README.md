
# Short-term Memory Active Learning (SMAL).
As active learning campaigns progress, it has been observed that performance can decrease after a "turning point" of maximum performance. In standard active learning, data is continuously added in a linear fashion. It is hypothesized that this could be problematic since selection functions are likely imperfect, especially during early stages of active learning campaigns where data is limited leading to a poorer understanding of a particular domain. SMAL was developed to augment standard active learning approaches by implementing backward forgetting of training data based on various measures of sample utility. Forgetting data leads to restricted training set sizes making models more compact and less biased, while leading to equivalent or improved overall performance. Additionally, the re-integration of prior experimental data reduces labeling costs, and enhances training set diversity and quality.

This package was built on [MolALKit](https://github.com/RekerLab/MolALKit).

## Installation
```commandline
pip install numpy==1.22.3 git+https://gitlab.com/Xiangyan93/graphdot.git@feature/xy git+https://github.com/bp-kelley/descriptastorus git+https://github.com/Xiangyan93/chemprop.git@molalkit
pip install mgktools molalkit
```

## Example
Experiment: Applying SMAL to the HIV dataset with the Maximum Out-of-Bag Uncertainty forgetting protocol

Additional parameters: 20% introduced error, 50:50 scaffold random train:test split, 1554 forget size (training set size, when to begin forgetting)

```commandline
python3 SMAL.py --data_public hiv --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride 1 --seed 3 --save_dir hiv-RF-Morgan-scaffold_random-explorative-max_oob_uncertainty-20-3 --n_jobs 1 --forget_size 1554 --forget_protocol max_oob_uncertainty --error_rate 0.2
```

## Active Learning and Other Usage
More information can be found at [MolALKit](https://github.com/RekerLab/MolALKit).
