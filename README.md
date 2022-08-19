# ActiveLearningBenchmark
Benchmark for molecular active learning.

## Installation
```commandline
conda env create -f environment.yml
conda activate alb
```
[GPU-enabled PyTorch](https://pytorch.org/get-started/locally/).
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
## Usage. BACE and freesolv datasets as examples.
### Random Forest on Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
```

### Gaussian Process Regression on Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionValue_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
```

Use posterior uncertainty for classification.
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_DotProductKerneL_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
```

### Multi-layer perceptron (MLP) on Morgan fingerprints
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/MLP_BinaryClassification_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/MLP_Regression_MVE_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/MLP_Regression_Evidential_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
```

### D-MPNN
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/DMPNN_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_Regression_MVE_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_Regression_Evidential_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
```

### D-MPNN + rdkit features
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_BinaryClassification_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_Regression_MVE_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/DMPNN_RDKIT_Regression_Evidential_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
```

### GPR-MGK
```commandline
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test --n_jobs 6
python3 ActiveLearning.py --data_public bace --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionValue_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test --n_jobs 6
python3 ActiveLearning.py --data_public freesolv --metrics rmse mae r2 --learning_type explorative --model_config_selector model_config/GaussianProcessRegressionUncertainty_MarginalizedGraphKerneL_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test
```

### Yoked Learning
```commandline
python3 ActiveLearning.py --data_path alb/data/bace.csv --pure_columns mol --target_columns Class --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test --model_config_evaluator model_config/MLP_BinaryClassification_Morgan_Config
```
