datasets=(2C9 2D6 3A4 3CL BACE BBBM BIOAVAILABILITY CARCINOGENS CLINTOX DILI HERG HIA PGP SKIN)
ntasks=1
n_stride=1
for((i=0;i<14;++i))
do
	break
	dataset=${datasets[$i]}
	file=data/$dataset/"$dataset"_training.csv
	echo $dataset
	for((j=0;j<20;++j))
	do
		if [[ $dataset == "HIA" ]]
		then
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector rf_config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-rf-$j --n_jobs $ntasks --model_config_extra_evaluators mlp_config data/$dataset/mlp_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector lr_config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-lr-$j --n_jobs $ntasks --model_config_extra_evaluators mlp_config data/$dataset/mlp_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector mlp_config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-mlp-default-$j --n_jobs $ntasks --model_config_extra_evaluators data/$dataset/mlp_config rf_config lr_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector data/$dataset/mlp_config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-mlp-opt-$j --n_jobs $ntasks --model_config_extra_evaluators mlp_config rf_config lr_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type passive --model_config_selector rf_config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-passive-$j --n_jobs $ntasks --model_config_extra_evaluators data/$dataset/mlp_config mlp_config rf_config lr_config
		else
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector rf_config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-rf-$j --n_jobs $ntasks --model_config_extra_evaluators mlp_config data/$dataset/mlp_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector lr_config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-lr-$j --n_jobs $ntasks --model_config_extra_evaluators mlp_config data/$dataset/mlp_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector mlp_config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-mlp-default-$j --n_jobs $ntasks --model_config_extra_evaluators data/$dataset/mlp_config rf_config lr_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector data/$dataset/mlp_config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-mlp-opt-$j --n_jobs $ntasks --model_config_extra_evaluators mlp_config rf_config lr_config
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type passive --model_config_selector rf_config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir $dataset-passive-$j --n_jobs $ntasks --model_config_extra_evaluators data/$dataset/mlp_config mlp_config rf_config lr_config
		fi
	done
done
# Only select samples, no performance evaluation.
for((i=0;i<14;++i))
do
	dataset=${datasets[$i]}
	file=data/$dataset/"$dataset"_training.csv
	echo $dataset
	for((j=0;j<3;++j))
	do
		if [[ $dataset == "HIA" ]]
		then
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector rf_config                --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-rf-$j          --n_jobs $ntasks --no_eval  
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector lr_config                --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-lr-$j          --n_jobs $ntasks --no_eval  
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector mlp_config               --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-mlp-default-$j --n_jobs $ntasks --no_eval  
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector data/$dataset/mlp_config --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-mlp-opt-$j     --n_jobs $ntasks --no_eval  
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type passive     --model_config_selector rf_config                --split_type scaffold_random --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-passive-$j     --n_jobs $ntasks --no_eval  
		else
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector rf_config                --split_type scaffold_order  --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-rf-$j          --n_jobs $ntasks --no_eval 
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector lr_config                --split_type scaffold_order  --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-lr-$j          --n_jobs $ntasks --no_eval 
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector mlp_config               --split_type scaffold_order  --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-mlp-default-$j --n_jobs $ntasks --no_eval 
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector data/$dataset/mlp_config --split_type scaffold_order  --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-mlp-opt-$j     --n_jobs $ntasks --no_eval 
			python3 ActiveLearning.py --data_path $file --pure_columns smiles --target_columns label --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type passive     --model_config_selector rf_config                --split_type scaffold_order  --split_sizes 0.5 0.5 --evaluate_stride $n_stride --seed $j --save_dir no-eval-$dataset-passive-$j     --n_jobs $ntasks --no_eval 
		fi
	done
done

