
# SMAL: Short-term Memory Active Learning.
This package implements forgetting data during active learning to reduce computational and experimental cost, while ehancing performance and training set construction.

This package was built on [MolALKit](https://github.com/RekerLab/MolALKit).

## Installation
```commandline
conda env create -f environment.yml
conda activate molalkit
```

## Example
Experiment: Applying SMAL to the HIV dataset with the Maximum Out-of-Bag Uncertainty forgetting protocol

Additional parameters: 20% introduced error, 50:50 scaffold random train:test split, 1554 forget size (training set size, when to begin forgetting)

```commandline
python3 SMAL.py --data_public hiv --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride 1 --seed 3 --save_dir hiv-RF-Morgan-scaffold_random-explorative-max_oob_uncertainty-20-3 --n_jobs 1 --forget_size 1554 --forget_protocol max_oob_uncertainty --error_rate 0.2
```

## Active Learning and Other Usage
More information can be found at [MolALKit](https://github.com/RekerLab/MolALKit).
