def available_classifiers():
    import autosklearn.pipeline.components.classification
    return list(autosklearn.pipeline.components.classification._classifiers.keys()) + \
           list(autosklearn.pipeline.components.classification._addons.components.keys())

def available_regressors():
    import autosklearn.pipeline.components.regression
    return list(autosklearn.pipeline.components.regression._regressors.keys()) + \
           list(autosklearn.pipeline.components.regression._addons.components.keys())

def available_feature_preprocessing():
    import autosklearn.pipeline.components.feature_preprocessing
    return list(autosklearn.pipeline.components.feature_preprocessing._preprocessors.keys()) + \
           list(autosklearn.pipeline.components.feature_preprocessing._addons.components.keys())

