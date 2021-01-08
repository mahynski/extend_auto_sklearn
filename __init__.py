"""Import module."""
__all__ = ["feature_preprocessing", "classifier", "regressor", "utils"]

import inspect

import autosklearn.pipeline.components.classification
import autosklearn.pipeline.components.feature_preprocessing
import autosklearn.pipeline.components.regression

from . import classifier, feature_preprocessing, regressor, utils

new_feature_preprocessing = [
    m[1]
    for m in inspect.getmembers(feature_preprocessing, inspect.isclass)
    if m[1].__module__ == __name__ + ".feature_preprocessing"
]
new_classifiers = [
    m[1]
    for m in inspect.getmembers(classifier, inspect.isclass)
    if m[1].__module__ == __name__ + ".classifier"
]
new_regressors = [
    m[1]
    for m in inspect.getmembers(regressor, inspect.isclass)
    if m[1].__module__ == __name__ + ".regressor"
]

for f in new_feature_preprocessing:
    autosklearn.pipeline.components.feature_preprocessing.add_preprocessor(f)

for c in new_classifiers:
    autosklearn.pipeline.components.classification.add_classifier(c)

for r in new_regressors:
    autosklearn.pipeline.components.regression.add_regressor(r)
