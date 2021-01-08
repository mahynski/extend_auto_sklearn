"""
Extend auto-sklearn to include additional regressors.

@author:nam
"""
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import (
    DENSE,
    PREDICTIONS,
    SIGNED_DATA,
    SPARSE,
    UNSIGNED_DATA,
)
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)


class PLSRegression(AutoSklearnRegressionAlgorithm):
    """PLS regression."""

    def __init__(self, n_components, scale):
        """Initialize the class."""
        self.n_components = n_components
        self.scale = scale
        self.estimator = None

    def fit(self, X, y):
        """Fit the regressor."""
        self.n_components = int(self.n_components)
        self.scale = bool(self.scale)

        import sklearn.cross_decomposition

        self.estimator = sklearn.cross_decomposition.PLSRegression(
            n_components=self.n_components,
            scale=self.scale,
            max_iter=1000,
            tol=1.0e-6,
            copy=True,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Predict after fit."""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of this regressor."""
        return {
            "shortname": "PLS Regression",
            "name": "Partial Least Squares Regression",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Get the configuration space used for hyperparameter searching."""
        cs = ConfigurationSpace()
        n_components = UniformIntegerHyperparameter(
            name="n_components", lower=1, upper=30, default_value=10
        )
        scale = CategoricalHyperparameter(
            name="scale", choices=[True, False], default_value=True
        )
        cs.add_hyperparameters([n_components, scale])

        return cs


class MLPRegressor(AutoSklearnRegressionAlgorithm):
    """MLP regressor."""

    def __init__(
        self,
        hidden_layer_depth,
        num_nodes_per_layer,
        activation,
        alpha,
        solver,
        random_state=None,
    ):
        """Initialize the class."""
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation = activation
        self.alpha = alpha
        self.solver = solver
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        """Fit the regressor."""
        self.num_nodes_per_layer = int(self.num_nodes_per_layer)
        self.hidden_layer_depth = int(self.hidden_layer_depth)
        self.alpha = float(self.alpha)

        import sklearn.neural_network

        hidden_layer_sizes = tuple(
            self.num_nodes_per_layer for i in range(self.hidden_layer_depth)
        )
        self.estimator = sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            solver=self.solver,
            random_state=self.random_state,
            max_iter=1000,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Predict after fit."""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of this regressor."""
        return {
            "shortname": "MLP Regressor",
            "name": "MLP Regressor",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": False,
            "input": (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Get the configuration space used for hyperparameter searching."""
        cs = ConfigurationSpace()
        hidden_layer_depth = UniformIntegerHyperparameter(
            name="hidden_layer_depth", lower=1, upper=5, default_value=1
        )
        num_nodes_per_layer = UniformIntegerHyperparameter(
            name="num_nodes_per_layer", lower=3, upper=20, default_value=5
        )
        activation = CategoricalHyperparameter(
            name="activation",
            choices=["identity", "logistic", "tanh", "relu"],
            default_value="relu",
        )
        alpha = UniformFloatHyperparameter(
            name="alpha",
            lower=0.0001,
            upper=1.0,
            default_value=0.0001,
            log=True,
        )
        solver = CategoricalHyperparameter(
            name="solver",
            choices=["lbfgs", "sgd", "adam"],
            default_value="lbfgs",
        )
        cs.add_hyperparameters(
            [
                hidden_layer_depth,
                num_nodes_per_layer,
                activation,
                alpha,
                solver,
            ]
        )
        return cs


class OrthogonalMatchingPursuit(AutoSklearnRegressionAlgorithm):
    """Orthogonal matching pursuit regressor."""

    def __init__(self, n_nonzero_coefs, fit_intercept):
        """Initialize the class."""
        self.n_nonzero_coefs = n_nonzero_coefs
        self.fit_intercept = fit_intercept

        self.estimator = None

    def fit(self, X, y):
        """Fit the regressor."""
        self.fit_intercept = bool(self.fit_intercept)
        self.n_nonzero_coefs = int(self.n_nonzero_coefs)

        import sklearn.linear_model

        self.estimator = sklearn.linear_model.OrthogonalMatchingPursuit(
            n_nonzero_coefs=self.n_nonzero_coefs,
            tol=None,
            fit_intercept=self.fit_intercept,
            normalize=False,
            precompute="auto",
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Predict after fit."""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of this regressor."""
        return {
            "shortname": "OMP",
            "name": "Orthogonal Matching Pursuit",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Get the configuration space used for hyperparameter searching."""
        cs = ConfigurationSpace()
        fit_intercept = CategoricalHyperparameter(
            name="fit_intercept", choices=[True, False], default_value=True
        )

        n_nonzero_coefs = UniformIntegerHyperparameter(
            name="n_nonzero_coefs", lower=1, upper=30, default_value=10
        )

        cs.add_hyperparameters([fit_intercept, n_nonzero_coefs])

        return cs


class ElasticNet(AutoSklearnRegressionAlgorithm):
    """Elastic net regressor."""

    def __init__(self, alpha, l1_ratio, fit_intercept, selection):
        """Initialize the class."""
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.selection = selection
        self.random_state = None
        self.estimator = None

    def fit(self, X, y):
        """Fit the regressor."""
        self.fit_intercept = bool(self.fit_intercept)
        self.alpha = float(self.alpha)
        self.l1_ratio = float(self.l1_ratio)

        import sklearn.linear_model

        self.estimator = sklearn.linear_model.ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            normalize=False,
            precompute=False,
            max_iter=5000,
            copy_X=True,
            tol=1.0e-4,
            warm_start=False,
            positive=False,
            random_state=self.random_state,
            selection=self.selection,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Predict after fit."""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of this regressor."""
        return {
            "shortname": "Elastic Net",
            "name": "Elastic Net Linear Regression",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": False,
            "input": (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Get the configuration space used for hyperparameter searching."""
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1.0e-3, upper=1.0e3, default_value=1.0, log=True
        )
        l1_ratio = UniformFloatHyperparameter(
            name="l1_ratio", lower=0.01, upper=1.0, default_value=0.5
        )
        fit_intercept = CategoricalHyperparameter(
            name="fit_intercept", choices=[True, False], default_value=True
        )
        selection = CategoricalHyperparameter(
            name="selection",
            choices=["cyclic", "random"],
            default_value="cyclic",
        )

        cs.add_hyperparameters([alpha, l1_ratio, fit_intercept, selection])

        return cs


class LinearRegression(AutoSklearnRegressionAlgorithm):
    """Linear regressor."""

    def __init__(self, fit_intercept, normalize):
        """Initialize the class."""
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.estimator = None

    def fit(self, X, y):
        """Fit the regressor."""
        self.fit_intercept = bool(self.fit_intercept)

        import sklearn.linear_model

        self.estimator = sklearn.linear_model.LinearRegression(
            fit_intercept=self.fit_intercept, normalize=False, copy_X=True
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Predict after fit."""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of this regressor."""
        return {
            "shortname": "Lin Reg",
            "name": "Linear Regression",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Get the configuration space used for hyperparameter searching."""
        cs = ConfigurationSpace()
        fit_intercept = CategoricalHyperparameter(
            name="fit_intercept", choices=[True, False], default_value=True
        )
        cs.add_hyperparameters([fit_intercept])

        return cs


class KernelRidgeRegression(AutoSklearnRegressionAlgorithm):
    """Kernel ridge regressor."""

    def __init__(self, alpha, kernel, gamma, degree, random_state=None):
        """Initialize the class."""
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        """Fit the regressor."""
        self.alpha = float(self.alpha)
        self.gamma = float(self.gamma)
        self.degree = int(self.degree)

        import sklearn.kernel_ridge

        self.estimator = sklearn.kernel_ridge.KernelRidge(
            alpha=self.alpha,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        """Predict after fit."""
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        """Get the properties of this regressor."""
        return {
            "shortname": "KRR",
            "name": "Kernel Ridge Regression",
            "handles_regression": True,
            "handles_classification": False,
            "handles_multiclass": False,
            "handles_multilabel": False,
            "handles_multioutput": True,
            "is_deterministic": True,
            "input": (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
            "output": (PREDICTIONS,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        """Get the configuration space used for hyperparameter searching."""
        cs = ConfigurationSpace()
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=10 ** -5, upper=1, log=True, default_value=0.1
        )
        kernel = CategoricalHyperparameter(
            name="kernel",
            choices=[
                "linear",
                "rbf",
                "sigmoid",
                "polynomial",
                "cosine",
                "laplacian",
            ],
            default_value="linear",
        )
        gamma = UniformFloatHyperparameter(
            name="gamma", lower=0.00001, upper=1, default_value=0.1, log=True
        )
        degree = UniformIntegerHyperparameter(
            name="degree", lower=2, upper=5, default_value=3
        )
        cs.add_hyperparameters([alpha, kernel, gamma, degree])

        return cs
