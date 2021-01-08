"""
Additional utilities.

@author:nam
"""
import autosklearn.pipeline.components.classification
import autosklearn.pipeline.components.feature_preprocessing
import autosklearn.pipeline.components.regression


def available_classifiers():
    """List all available classifiers."""
    return list(
        autosklearn.pipeline.components.classification._classifiers.keys()
    ) + list(
        autosklearn.pipeline.components.classification._addons.components.keys()
    )


def available_regressors():
    """List all available regressors."""
    return list(
        autosklearn.pipeline.components.regression._regressors.keys()
    ) + list(
        autosklearn.pipeline.components.regression._addons.components.keys()
    )


def available_feature_preprocessing():
    """List all available feature preprocessors."""
    return list(
        autosklearn.pipeline.components.feature_preprocessing._preprocessors.keys()
    ) + list(
        autosklearn.pipeline.components.feature_preprocessing._addons.components.keys()
    )
