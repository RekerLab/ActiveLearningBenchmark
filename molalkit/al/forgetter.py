#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from alb.models.random_forest.RandomForestClassifier import RFClassifier
from alb.models.gaussian_process.GaussianProcessRegressor import GPRegressor
from alb.al.selection_method import get_topn_idx


class BaseForgetter(ABC):
    def __init__(self, batch_size: int = 1, forget_size: int = 1, forget_cutoff: float = None):
        self.batch_size = batch_size
        self.forget_size = forget_size
        self.forget_cutoff = forget_cutoff

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[List[int], List[float]]:
        pass

    @property
    @abstractmethod
    def info(self) -> str:
        pass


class BaseRandomForgetter(BaseForgetter, ABC):
    """Base Forgetter that uses random seed."""
    def __init__(self, batch_size: int = 1, forget_size: int = 1, forget_cutoff: float = None, seed: int = 0):
        super().__init__(batch_size=batch_size, forget_size=forget_size, forget_cutoff=forget_cutoff)
        np.random.seed(seed)


class RandomForgetter(BaseRandomForgetter):
    def __call__(self, data, batch_size: int = 1) -> Tuple[List[int], None]:
        assert batch_size < len(data)
        return np.random.choice(list(range(len(data))), batch_size, replace=False).tolist(), None

    @property
    def info(self) -> str:
        return 'RandomForgetter'


class FirstForgetter(BaseForgetter):
    def __call__(self, data, batch_size: int = 1) -> Tuple[List[int], None]:
        assert batch_size < len(data)
        return list(range(batch_size)), None

    @property
    def info(self) -> str:
        return 'FirstForgetter'


class MinOOBUncertaintyForgetter(BaseRandomForgetter):
    def __call__(self, model: RFClassifier, data, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        assert model.oob_score is True
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size, target='min')
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinOOBUncertaintyForgetter'


class MinOOBUncertaintyCorrectForgetter(BaseRandomForgetter):
    """Forget samples with lowest out-of-bag (OOB) uncertainty that were correctly predicted OOB"""
    def __call__(self, model: RFClassifier, data, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        assert model.oob_score is True
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        y_pred = np.argmax(y_oob_proba, axis=1)
        y = data.y
        incorrect = np.where(y_pred != y)[0]
        # since want min correct, set incorrect predictions to 1.1 (will always be most uncertain and never chosen)
        y_oob_uncertainty[incorrect] = 1.1
        # select the top-n correct points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size, target='min')
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinOOBUncertaintyCorrectForgetter'
        

class MinOOBUncertaintyIncorrectForgetter(BaseRandomForgetter):
    """Forget samples with lowest out-of-bag (OOB) uncertainty that were incorrectly predicted OOB"""
    def __call__(self, model: RFClassifier, data, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        assert model.oob_score is True
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        y_pred = np.argmax(y_oob_proba, axis=1)
        y = data.y
        correct = np.where(y_pred == y)[0]
        # since want min incorrect, set correct predictions to 1.1 (will always be most uncertain and never chosen)
        y_oob_uncertainty[correct] = 1.1
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size, target='min')
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinOOBUncertaintyInorrectForgetter'


class MaxOOBUncertaintyForgetter(BaseRandomForgetter):
    def __call__(self, model: RFClassifier, data, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the highest out-of-bag (OOB) uncertainty.

        Parameters
        ----------
        model: Only random forest classifier is supported due to efficient OOB uncertainty calculation.
        data: The dataset to forget.
        batch_size: The number of samples to forget.

        Returns
        -------
        The index of samples to forget.
        """
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size)
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MaxOOBUncertaintyForgetter'

class MaxOOBUncertaintyCorrectForgetter(BaseRandomForgetter):
    """Forget samples with highest out-of-bag (OOB) uncertainty that were correctly predicted OOB"""
    def __call__(self, model: RFClassifier, data, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        assert model.oob_score is True
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        y_pred = np.argmax(y_oob_proba, axis=1)
        y = data.y
        incorrect = np.where(y_pred != y)[0]
        # since want max correct, set incorrect predictions to -0.1 (will always be least uncertain and never chosen)
        y_oob_uncertainty[incorrect] = -0.1
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size)
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MaxOOBUncertaintyCorrectForgetter'
        

class MaxOOBUncertaintyIncorrectForgetter(BaseRandomForgetter):
    """Forget samples with highest out-of-bag (OOB) uncertainty that were incorrectly predicted OOB"""
    def __call__(self, model: RFClassifier, data, batch_size: int = 1) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        assert model.oob_score is True
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        y_oob_uncertainty = (0.25 - np.var(y_oob_proba, axis=1)) * 4
        y_pred = np.argmax(y_oob_proba, axis=1)
        y = data.y
        correct = np.where(y_pred == y)[0]
        # since want max incorrect, set correct predictions to -0.1 (will always be least uncertain and never chosen)
        y_oob_uncertainty[correct] = -0.1
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(y_oob_uncertainty, n=batch_size)
        acquisition = y_oob_uncertainty[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MaxOOBUncertaintyInorrectForgetter'


class MinOOBErrorForgetter(BaseRandomForgetter):
    def __call__(self, model: RFClassifier, data, batch_size: int = 1, cutoff: float = None, **kwargs
                 ) -> Tuple[List[int], List[float]]:
        assert batch_size < len(data)
        assert isinstance(model, RFClassifier)
        y_oob_proba = model.oob_decision_function_
        # uncertainty calculation, normalized into 0 to 1
        oob_error = np.absolute(y_oob_proba[:, 1] - data.y)
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(oob_error, n=batch_size, target='min', cutoff=cutoff)
        acquisition = oob_error[np.array(forgotten_idx)].tolist() if forgotten_idx else []
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinOOBErrorForgetter'


class MinLOOErrorForgetter(BaseRandomForgetter):
    def __call__(self, model: GPRegressor, data, batch_size: int = 1, cutoff: float = None, **kwargs
                 ) -> Tuple[List[int], List[float]]:
        """ Forget the samples with the lowest Leave-one-out cross-validation (LOOCV) error.
        Parameters
        ----------
        model: Only Gaussian process regressor is supported due to efficient LOOCV of GPR.
        data: The dataset to forget.
        batch_size: The number of samples to forget.
        cutoff: The cutoff value of LOOCV error. Only samples with LOOCV error lower than cutoff will be forgot.

        Returns
        -------
        The index and the acquisition value of samples to forget.
        """
        assert batch_size < len(data)
        assert isinstance(model, GPRegressor)
        y_loocv = model.predict_loocv(data.X, data.y, return_std=False)
        # uncertainty calculation, normalized into 0 to 1
        loo_error = np.absolute(y_loocv - data.y)
        # select the top-n points with least uncertainty
        forgotten_idx = get_topn_idx(loo_error, n=batch_size, target='min', cutoff=cutoff)
        acquisition = loo_error[np.array(forgotten_idx)].tolist()
        return forgotten_idx, acquisition

    @property
    def info(self) -> str:
        return 'MinLOOErrorForgetter'
