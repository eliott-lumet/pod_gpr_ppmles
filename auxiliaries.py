#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module which contains auxiliary functions and classes required by the POD--GPR
surrogate model.

@author: lumet
"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np

from numbers import Integral

from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, clone, MetaEstimatorMixin
from sklearn.base import RegressorMixin, is_classifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_is_fitted,
    has_fit_parameter,
    _check_method_params,  # _check_fit_params for scikit-learn older versions (<= 1.2.1)
)
from sklearn.utils.parallel import delayed, Parallel
from sklearn.utils._param_validation import HasMethods

# =============================================================================
# Functions
# =============================================================================
def min_max_normalize(x, xmin, xmax, s=1):
    """
    Rescaling a set of data from [xmin, xmax] to [0., 1.].
    """
    return s * (x - xmin)/(xmax - xmin) 

def log_transform(y, threshold):
    """
    Apply a log transformation with threshold to a dataset.
    """
    return np.log(y + threshold)

def log_inverse_transform(y, threshold):
    """
    Apply a the inverse transformation for data that was converted with the 
    log_transform function.
    """
    return np.exp(y) - threshold

# =============================================================================
# Centered scaler function with log preprocessing
# =============================================================================
class LogCenteredScaler():
    """
    Class used to center and normalize after a log-transform dataset, it is 
    used before and after a POD-log projection for the MUST surrogate.
    """
    def __init__(self, log_cut=1e-4):
        self.mean = None
        self.weights = None
        self.n_samples = None
        self.log_cut = log_cut
        
    def fit(self, data, weights=None):
        self.mean = np.mean(log_transform(data, self.log_cut), axis=0)
        self.weights = weights
        self.n_samples = data.shape[0]
        return self
        
    def transform(self, data):
        if self.weights is None:
            T = (log_transform(data, self.log_cut) - self.mean)
        else:
            T = (log_transform(data, self.log_cut) - self.mean)*np.sqrt(self.weights/np.sum(self.weights)).T

        return T
    
    def inverse_transform(self, data):
        if self.weights is None:
            out = log_inverse_transform(data + self.mean, self.log_cut)
        else:
            out = log_inverse_transform(np.sqrt(np.sum(self.weights)/self.weights)*data + self.mean, self.log_cut)

        return out

# =============================================================================
# Multiple surrogate regressor wrapped modified from scikit-learn.
# =============================================================================
def _fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


def _partial_fit_estimator(
    estimator, X, y, classes=None, sample_weight=None, first_time=True
):
    if first_time:
        estimator = clone(estimator)

    if sample_weight is not None:
        if classes is not None:
            estimator.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        else:
            estimator.partial_fit(X, y, sample_weight=sample_weight)
    else:
        if classes is not None:
            estimator.partial_fit(X, y, classes=classes)
        else:
            estimator.partial_fit(X, y)
    return estimator


def _available_if_estimator_has(attr):
    """Return a function to check if `estimator` or `estimators_` has `attr`.
    Helper for Chain implementations.
    """

    def _check(self):
        return hasattr(self.estimator, attr) or all(
            hasattr(est, attr) for est in self.estimators_
        )

    return available_if(_check)


class _MultiOutputEstimator(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict", "sample_y"])],
        "n_jobs": [Integral, None],
    }

    @abstractmethod
    def __init__(self, estimator, *, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs

    @_available_if_estimator_has("partial_fit")
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """Incrementally fit a separate model for each class output.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.
        classes : list of ndarray of shape (n_outputs,), default=None
            Each array is unique classes for one output in str/int.
            Can be obtained via
            ``[np.unique(y[:, i]) for i in range(y.shape[1])]``, where `y`
            is the target matrix of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that `y` doesn't need to contain all labels in `classes`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        first_time = not hasattr(self, "estimators_")

        if first_time:
            self._validate_params()

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support sample weights.")

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_partial_fit_estimator)(
                self.estimators_[i] if not first_time else self.estimator,
                X,
                y[:, i],
                classes[i] if classes is not None else None,
                sample_weight,
                first_time,
            )
            for i in range(y.shape[1])
        )

        if first_time and hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if first_time and hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.
            .. versionadded:: 0.23
        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        self._validate_params()

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support sample weights.")

        fit_params_validated = _check_method_params(X, fit_params)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator, X, y[:, i], sample_weight, **fit_params_validated
            )
            for i in range(y.shape[1])
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X, **predict_params):
        """Predict multi-output variable using model for each target variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
            Note: Separate models are generated for each predictor.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X, **predict_params) for e in self.estimators_
        )

        return np.asarray(y).T

    def sample_y(self, X, **sample_params):
        """Draw samples from Gaussian process and evaluate at X.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.

        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.

        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "sample_y"):
            raise ValueError("The base estimator should implement a sample_y method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.sample_y)(X, **sample_params) for e in self.estimators_
        )

        return np.asarray(y).T

    def _more_tags(self):
        return {"multioutput_only": True}


class MultiOutputRegressor(RegressorMixin, _MultiOutputEstimator):
    """Multi target regression.
    This strategy consists of fitting one regressor per target. This is a
    simple strategy for extending regressors that do not natively support
    multi-target regression.
    .. versionadded:: 0.18
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing :term:`fit` and :term:`predict`.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        :meth:`fit`, :meth:`predict` and :meth:`partial_fit` (if supported
        by the passed estimator) will be parallelized for each target.
        When individual estimators are fast to train or predict,
        using ``n_jobs > 1`` can result in slower performance due
        to the parallelism overhead.
        ``None`` means `1` unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all available processes / threads.
        See :term:`Glossary <n_jobs>` for more details.
        .. versionchanged:: 0.20
            `n_jobs` default changed from `1` to `None`.
    Attributes
    ----------
    estimators_ : list of ``n_output`` estimators
        Estimators used for predictions.
    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying `estimator` exposes such an attribute when fit.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0
    See Also
    --------
    RegressorChain : A multi-label model that arranges regressions into a
        chain.
    MultiOutputClassifier : Classifies each output independently rather than
        chaining.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_linnerud
    >>> from sklearn.multioutput import MultiOutputRegressor
    >>> from sklearn.linear_model import Ridge
    >>> X, y = load_linnerud(return_X_y=True)
    >>> regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X, y)
    >>> regr.predict(X[[0]])
    array([[176..., 35..., 57...]])
    """

    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    @_available_if_estimator_has("partial_fit")
    def partial_fit(self, X, y, sample_weight=None):
        """Incrementally fit the model to data, for each output variable.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        super().partial_fit(X, y, sample_weight=sample_weight)