#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List, Dict, Optional, Callable
import numpy as np
import pandas as pd
from mgktools.features_mol.features_generators import FeaturesGenerator
from molalkit.models.utils import get_kernel, get_model


def get_features_generators_from_config(model_config: Dict) -> Optional[List[List[FeaturesGenerator]]]:
    fingerprints_class = model_config.get("fingerprints_class")
    radius = model_config.get("radius")
    num_bits = model_config.get("num_bits")
    atomInvariantsGenerator = model_config.get("atomInvariantsGenerator")
    if fingerprints_class is None:
        return None
    else:
        return [FeaturesGenerator(features_generator_name=fc,
                                  radius=radius,
                                  num_bits=num_bits,
                                  atomInvariantsGenerator=atomInvariantsGenerator) for fc in fingerprints_class]


def get_kernel_from_config(model_config: Dict, dataset, kernel_pkl_path) -> Callable:
    return get_kernel(
        graph_kernel_type=model_config.get("graph_kernel_type"),
        mgk_files=model_config.get("mgk_files"),
        features_kernel_type=model_config.get("features_kernel_type"),
        features_hyperparameters=model_config.get("features_hyperparameters"),
        features_hyperparameters_file=model_config.get("features_hyperparameters_file"),
        dataset=dataset,
        kernel_pkl_path=kernel_pkl_path,
    )


def get_model_from_config(model_config: Dict, dataset, task_type, save_dir,
                          data_path, smiles_columns, targets_columns, 
                          features_generators, kernel,
                          n_jobs, seed, logger) -> Callable:
    try:
        features_size = dataset.features_size()
    except:
        features_size = None
    return get_model(
        data_format=model_config["data_format"],
        task_type=task_type,
        model=model_config.get("model"),
        save_dir=save_dir,
        data_path=data_path,
        smiles_columns=smiles_columns,
        target_columns=targets_columns,
        loss_function=model_config.get("loss_function"),
        multiclass_num_classes=model_config.get("loss_function") or 3,
        features_generator=features_generators,
        no_features_scaling=model_config.get("no_features_scaling") or False,
        features_only=model_config.get("features_only") or False,
        features_size=features_size,
        epochs=model_config.get("epochs") or 30,
        depth=model_config.get("depth") or 3,
        hidden_size=model_config.get("hidden_size") or 300,
        ffn_num_layers=model_config.get("ffn_num_layers") or 2,
        ffn_hidden_size=model_config.get("ffn_hidden_size"),
        dropout=model_config.get("dropout") or 0.0,
        batch_size=model_config.get("batch_size") or 50,
        ensemble_size=model_config.get("ensemble_size") or 1,
        number_of_molecules=model_config.get("number_of_molecules") or len(smiles_columns),
        mpn_shared=model_config.get("mpn_shared") or False,
        atom_messages=model_config.get("atom_messages") or False,
        undirected=model_config.get("undirected") or False,
        class_balance=model_config.get("class_balance") or False,
        checkpoint_dir=model_config.get("checkpoint_dir"),
        checkpoint_frzn=model_config.get("checkpoint_frzn"),
        frzn_ffn_layers=model_config.get("frzn_ffn_layers") or 0,
        freeze_first_only=model_config.get("freeze_first_only") or False,
        mpn_path=model_config.get("mpn_path"),
        freeze_mpn=model_config.get("freeze_mpn") or False,
        continuous_fit=model_config.get("continuous_fit") or False,
        kernel=kernel,
        uncertainty_type=model_config.get("uncertainty_type"),
        alpha=model_config.get("alpha"),
        C=model_config.get("C"),
        booster=model_config.get("booster"),
        n_estimators=model_config.get("n_estimators") or 100,
        max_depth=model_config.get("max_depth"),
        learning_rate=model_config.get("learning_rate") or 0.1,
        n_jobs=n_jobs,
        seed=seed,
        logger=logger
    )


def read_df(path, task_type, target_columns) -> pd.DataFrame:
    df = pd.read_csv(path)
    if task_type in ["binary", "multiclass"]:
        for target_column in target_columns:
            if df[target_column].dtype == "float":
                df[target_column] = df[target_column].astype(int)
    return df


def add_error_rate_to_labels(df, error_rate, target_column):
    """ Randomly flip the labels of a portion of the data."""
    error_index = np.random.choice(df.index, int(error_rate * len(df)), replace=False)
    df["flip_label"] = False
    df.loc[error_index, target_column] ^= 1
    df.loc[error_index, "flip_label"] = True


def apply_error_rate_to_id2datapoint(id2datapoint, df, target_column):
    assert "flip_label" in df.columns, "Error rate has not been applied to the dataframe."
    for i, row in df.iterrows():
        if row["flip_label"]:
            assert set([id2datapoint[row["uidx"]].targets[0], row[target_column]]) == {0, 1}
            id2datapoint[row["uidx"]].targets[0] = row[target_column]
        else:
            assert id2datapoint[row["uidx"]].targets[0] == row[target_column]
