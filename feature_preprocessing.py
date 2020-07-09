import autosklearn.classification
import autosklearn.pipeline.components.feature_preprocessing
from autosklearn.pipeline.components.base import AutoSklearnPreprocessingAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA
from autosklearn.util.common import check_none
from ConfigSpace.conditions import InCondition, AndConjunction, EqualsCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformIntegerHyperparameter, UniformFloatHyperparameter

class RFE(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, base_estimator, n_features_to_select, step,
                 rf_n_estimators=100, 
                 rf_criterion='gini', 
                 rf_max_depth=None, 
                 rf_class_weight='balanced_subsample', 
                 lr_C=1.0,
                 lr_penalty='l2',
                 lr_fit_intercept=True,
                 lr_intercept_scaling=1.0,
                 lr_class_weight='balanced',
                 svc_C=1.0,
                 svc_penalty='l2',
                 svc_class_weight='balanced',
                 svc_loss='squared_hinge',
                 svc_fit_intercept=True,
                 random_state=None
                ):
        
        self.base_estimator = base_estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.random_state = random_state
        self.preprocessor = None
        
        # For RF
        self.rf_n_estimators = rf_n_estimators
        self.rf_criterion = rf_criterion
        self.rf_max_depth = rf_max_depth
        self.rf_class_weight = rf_class_weight
        
        # For LR
        self.lr_C = lr_C
        self.lr_penalty = lr_penalty
        self.lr_fit_intercept = lr_fit_intercept
        self.lr_intercept_scaling = lr_intercept_scaling
        self.lr_class_weight = lr_class_weight
        
        # For SVC
        self.svc_C = svc_C
        self.svc_penalty = svc_penalty
        self.svc_class_weight = svc_class_weight
        self.svc_loss = svc_loss
        self.svc_fit_intercept = svc_fit_intercept
        
    def fit(self, X, y=None):
        if (self.base_estimator == 'rf'):
            import sklearn.ensemble.RandomForestClassifier as RandomForestClassifier
            self.rf_n_estimators = int(self.rf_n_estimators)
            self.rf_max_depth = int(self.rf_max_depth)
        
            base = RandomForestClassifier(n_estimators=self.rf_n_estimators,
                                          criterion=self.rf_criterion,
                                          max_depth=self.rf_max_depth,
                                          random_state=self.random_state,
                                          bootstrap=True,
                                          max_features='sqrt',
                                          class_weight=self.rf_class_weight)
        elif (self.base_estimator == 'lr'):
            import sklearn.linear_model.LogisticRegression as LogisticRegression
            self.lr_fit_intercept = bool(self.lr_fit_intercept)
            self.lr_intercept_scaling = float(self.lr_intercept_scaling)
            self.lr_C = float(self.lr_C)
            
            base = LogisticRegression(penalty=self.lr_penalty, 
                                      C=self.lr_C, 
                                      dual=False, 
                                      tol=1.0e-6, 
                                      random_state=self.random_state, 
                                      max_iter=1e3,
                                      intercept_scaling=self.lr_intercept_scaling,
                                      multi_class='auto', # To allow diff solvers and penalties to play nicely
                                      fit_intercept=self.lr_fit_intercept,
                                      class_weight=self.lr_class_weight)
        elif (self.base_estimator == 'svc'):
            import sklearn.svm.LinearSVC as LinearSVC
            self.svc_fit_intercept = bool(self.svc_fit_intercept)
            self.svc_C = float(self.svc_C)
            
            base = LinearSVC(C=self.svc_C, 
                             penalty=self.svc_penalty,
                             dual=False,
                             random_state=self.random_state, 
                             max_iter=1e5,
                             multi_class='ovr',
                             class_weight=self.svc_class_weight,
                             loss=self.svc_loss,
                             fit_intercept=self.svc_fit_intercept)
        else:
            raise ValueError('Unrecognized base estimator {}'.format(self.base_estimator))
        
        import sklearn.feature_selection
        self.preprocessor = \
            sklearn.feature_selection.RFE(
                estimator=base,
                n_features_to_select=int(self.n_features_to_select),
                step=self.step
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RFE',
                'name': 'Recursive Feature Elimination',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False, # if RNG seed specified, could be true?
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        conditions = []
        
        base_estimator = CategoricalHyperparameter(
            name="base_estimator", choices=['rf', 'lr', 'svc'], default_value='rf'
        )
        n_features_to_select = UniformIntegerHyperparameter(
            name="n_features_to_select", lower=1, upper=30, default_value=10
        )
        step = UniformFloatHyperparameter(
            name="step", lower=0.01, upper=0.25, default_value=0.1
        )
        
        # RF
        rf_n_estimators = UniformIntegerHyperparameter(
            name="rf_n_estimators", lower=1, upper=1000, default_value=100
        )
        rf_criterion = CategoricalHyperparameter(
            name="rf_criterion", choices=['gini', 'entropy'], default_value='gini'
        )
        rf_max_depth = UniformIntegerHyperparameter(
            name="rf_max_depth", lower=1, upper=100, default_value=10
        )
        rf_class_weight = CategoricalHyperparameter(
            name="rf_class_weight", choices=['balanced', 'balanced_subsample'], default_value='balanced_subsample'
        )
        conditions.append(EqualsCondition(rf_n_estimators, base_estimator, 'rf'))
        conditions.append(EqualsCondition(rf_criterion, base_estimator, 'rf'))
        conditions.append(EqualsCondition(rf_max_depth, base_estimator, 'rf'))
        conditions.append(EqualsCondition(rf_class_weight, base_estimator, 'rf'))
        
        # LR
        lr_C = UniformFloatHyperparameter(
            name="lr_C", lower=1.0e-3, upper=1.0e3, default_value=1.0, log=True
        )
        lr_penalty = CategoricalHyperparameter(
            name="lr_penalty", choices=['l1', 'l2'], default_value='l2'
        )
        lr_fit_intercept = CategoricalHyperparameter(
            name="lr_fit_intercept", choices=[True, False], default_value=True
        )
        lr_intercept_scaling = UniformFloatHyperparameter(
            name="lr_intercept_scaling", lower=1.0, upper=1.0e3, default_value=1.0
        )
        lr_class_weight = CategoricalHyperparameter(
            name="lr_class_weight", choices=['balanced', 'None'], default_value='balanced'
        )
        conditions.append(EqualsCondition(lr_C, base_estimator, 'lr'))
        conditions.append(EqualsCondition(lr_penalty, base_estimator, 'lr'))
        conditions.append(EqualsCondition(lr_fit_intercept, base_estimator, 'lr'))
        conditions.append(AndConjunction(
                            EqualsCondition(lr_intercept_scaling, base_estimator, 'lr'),
                            EqualsCondition(lr_intercept_scaling, lr_fit_intercept, True))
                         )
        conditions.append(EqualsCondition(lr_class_weight, base_estimator, 'lr'))
        
        # SVC
        svc_C = UniformFloatHyperparameter(
            name="svc_C", lower=1.0e-3, upper=1.0e3, default_value=1.0, log=True
        )
        svc_penalty = CategoricalHyperparameter(
            name="svc_penalty", choices=['l1', 'l2'], default_value='l2'
        )
        svc_class_weight = CategoricalHyperparameter(
            name="svc_class_weight", choices=['balanced', 'None'], default_value='balanced'
        )
        svc_loss = CategoricalHyperparameter(
            name="svc_loss", choices=['hinge', 'squared_hinge'], default_value='squared_hinge'
        )
        svc_fit_intercept = CategoricalHyperparameter(
            name="svc_fit_intercept", choices=[True, False], default_value=True
        )
        conditions.append(EqualsCondition(svc_C, base_estimator, 'svc'))
        conditions.append(EqualsCondition(svc_penalty, base_estimator, 'svc'))
        conditions.append(EqualsCondition(svc_class_weight, base_estimator, 'svc'))
        conditions.append(EqualsCondition(svc_loss, base_estimator, 'svc'))
        conditions.append(EqualsCondition(svc_fit_intercept, base_estimator, 'svc'))
        
        cs.add_hyperparameters([base_estimator, n_features_to_select, step,
                                rf_n_estimators, rf_criterion, rf_max_depth, rf_class_weight,
                                lr_C, lr_penalty, lr_fit_intercept, lr_intercept_scaling, lr_class_weight,
                                svc_C, svc_penalty, svc_class_weight, svc_loss, svc_fit_intercept
                               ])
        for c in conditions:
            cs.add_condition(c)
        
        return cs

class SparsePCA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, n_components, tol, alpha, ridge_alpha, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.tol = tol
        self.random_state = random_state # Could be deterministic if set to a value?
        self.preprocessor = None

    def fit(self, X, y=None):
        self.n_components = int(self.n_components)
        self.alpha = float(self.alpha)
        self.ridge_alpha = float(self.ridge_alpha)
        self.tol = float(self.tol)
        
        import sklearn.decomposition
        self.preprocessor = \
            sklearn.decomposition.SparsePCA(
                n_components=self.n_components,
                alpha=self.alpha,
                ridge_alpha=self.ridge_alpha,
                tol=self.tol,
                random_state=self.random_state,
                method='lars',
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'SparsePCA',
                'name': 'Sparse Principal Component Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': False, # if RNG seed specified, could be true?
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=1, upper=30, default_value=10
        )
        tol = UniformFloatHyperparameter(
            name="tol", lower=1.0e-12, upper=1, default_value=1.0e-8
        )
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1.0e-3, upper=1.0e3, default_value=1.0, log=True
        )
        ridge_alpha = UniformFloatHyperparameter(
            name="ridge_alpha", lower=1.0e-6, upper=1.0, default_value=1.0e-2, log=True
        )
        cs.add_hyperparameters([n_components, tol, alpha, ridge_alpha])
        
        return cs
    
class LDA(AutoSklearnPreprocessingAlgorithm):
    def __init__(self, solver, n_components, tol, shrinkage=None, random_state=None):
        self.solver = solver
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.tol = tol
        self.random_state = random_state
        self.preprocessor = None

    def fit(self, X, y=None):
        if check_none(self.shrinkage):
            self.shrinkage = None
        else:
            self.shrinkage = float(self.shrinkage)
        self.n_components = int(self.n_components)
        self.tol = float(self.tol)

        import sklearn.discriminant_analysis
        self.preprocessor = \
            sklearn.discriminant_analysis.LinearDiscriminantAnalysis(
                shrinkage=self.shrinkage,
                solver=self.solver,
                n_components=self.n_components,
                tol=self.tol,
            )
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'LDA',
                'name': 'Linear Discriminant Analysis',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True, # Was False in tutorial, I changed to True
                'handles_multilabel': False,
                'is_deterministic': True,
                'input': (DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (DENSE, UNSIGNED_DATA, SIGNED_DATA)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        solver = CategoricalHyperparameter(
            name="solver", choices=['svd', 'lsqr', 'eigen'], default_value='svd'
        )
        shrinkage = UniformFloatHyperparameter(
            name="shrinkage", lower=0.0, upper=1.0, default_value=0.5
        )
        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=1, upper=30, default_value=10
        )
        tol = UniformFloatHyperparameter(
            name="tol", lower=0.0001, upper=1, default_value=0.0001
        )
        cs.add_hyperparameters([solver, shrinkage, n_components, tol])
        shrinkage_condition = InCondition(shrinkage, solver, ['lsqr', 'eigen'])
        cs.add_condition(shrinkage_condition)
        
        return cs