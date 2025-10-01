# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 08:53:50 2020

@author: AliHaidar

This module defines a class `pspso` that performs hyper‑parameter
optimisation for a variety of machine learning algorithms using the
Particle Swarm Optimisation (PSO) algorithm.  The implementation has
been updated to import Keras components from the `tensorflow.keras`
namespace to improve compatibility with modern TensorFlow releases.
"""

import random
import time
import itertools as it
import numpy as np

# Attempt to import optional heavy dependencies.  These imports are
# deliberately wrapped in try/except blocks so that the module can be
# imported even if optional dependencies are not yet installed.  When
# a particular algorithm is used, the corresponding import is checked
# again and a clear error is raised if missing.
try:
    import lightgbm as lgb  # type: ignore
except ImportError:
    lgb = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except ImportError:
    xgb = None  # type: ignore

try:
    from sklearn.svm import SVC, SVR  # type: ignore
except ImportError:
    SVC = None  # type: ignore
    SVR = None  # type: ignore

# TensorFlow/Keras imports.  Use the bundled Keras API provided by
# TensorFlow.  These are loaded lazily in the MLP routines.
try:
    from tensorflow.keras.models import Sequential  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau  # type: ignore
    from tensorflow.keras.layers import Dense  # type: ignore
except ImportError:
    Sequential = None  # type: ignore
    EarlyStopping = None  # type: ignore
    ModelCheckpoint = None  # type: ignore
    TensorBoard = None  # type: ignore
    ReduceLROnPlateau = None  # type: ignore
    Dense = None  # type: ignore

# Attempt to import PySwarms for PSO optimisation.  If unavailable,
# `ps` will be set to `None` and an informative error will be raised
# when attempting to run optimisation routines.
try:
    import pyswarms as ps  # type: ignore
except ImportError:
    ps = None  # type: ignore
from sklearn.metrics import mean_squared_error, accuracy_score, auc, roc_curve


class pspso:
    """
    Particle Swarm Optimisation for hyper‑parameter search.

    The `pspso` class searches for algorithm parameters by using the
    particle swarm optimisation algorithm.  It supports a variety of
    estimators (currently XGBoost, Gradient Boosting Decision Trees,
    Support Vector Machines and Multi‑Layer Perceptrons) and both
    regression and binary classification tasks.  Results of the search
    (best parameters, cost, trained model, duration, etc.) are stored
    on the instance and accessible after fitting.
    """

    best_paricle_cost_ann = None
    best_model_ann = None
    best_history_ann = None
    best_particle_position_ann = None

    verbose = 0
    early_stopping = 20

    defaultparams = None  # contains the default parameters of the algorithm
    parameters = None  # contains the list of the parameters selected for optimisation
    paramdetails = None  # dictionary of input parameters
    rounding = None  # determines rounding for numerical parameters

    def __init__(self, estimator='xgboost', params=None, task="regression", score='rmse'):
        """Construct an instance of the `pspso` class.

        Parameters
        ----------
        estimator : str, optional
            One of 'xgboost', 'gbdt', 'mlp' or 'svm'.  Determines which
            underlying estimator's hyper‑parameters to optimise.

        params : dict or None, optional
            Dictionary specifying the search space for each parameter.  If
            `None`, a default search space is used (see
            :meth:`get_default_search_space`).

        task : str, optional
            Task type; either 'regression' or 'binary classification'.

        score : str, optional
            Fitness metric to minimise: 'rmse' for regression or 'acc'/'auc'
            for classification.
        """
        self.estimator = estimator
        self.task = task
        self.score = score
        self.cost = None
        self.pos = None
        self.model = None
        self.duration = None
        self.rmse = None
        self.optimizer = None
        # Read parameters and initialise search space
        (pspso.parameters,
         pspso.defaultparams,
         self.x_min,
         self.x_max,
         pspso.rounding,
         self.bounds,
         self.dimensions,
         pspso.paramdetails) = pspso.read_parameters(params, self.estimator, self.task)

    @staticmethod
    def get_default_search_space(estimator, task):
        """Create a dictionary of default parameter ranges.

        Parameters
        ----------
        estimator : str
            Name of the estimator ('mlp', 'xgboost', 'svm' or 'gbdt').

        task : str
            Task type: 'regression' or 'binary classification'.

        Returns
        -------
        dict
            Default search space mapping parameter names to either a list
            of categorical values or [min, max, decimal_places] for
            numerical ranges.
        """
        if estimator == 'xgboost':
            if task == 'binary classification':
                params = {
                    "learning_rate": [0.1, 0.3, 2],
                    "max_depth": [1, 10, 0],
                    "n_estimators": [2, 70, 0],
                    "subsample": [0.7, 1, 2],
                }
            else:
                params = {
                    "objective": ["reg:linear", "reg:tweedie", "reg:gamma"],
                    "learning_rate": [0.1, 0.3, 2],
                    "max_depth": [1, 10, 0],
                    "n_estimators": [2, 70, 0],
                    "subsample": [0.7, 1, 2],
                }
        elif estimator == 'gbdt':
            if task == 'binary classification':
                params = {
                    "learning_rate": [0.1, 0.3, 2],
                    "max_depth": [1, 10, 0],
                    "n_estimators": [2, 70, 0],
                    "subsample": [0.7, 1, 2],
                }
            else:
                params = {
                    "objective": ["tweedie", "regression"],
                    "learning_rate": [0.1, 0.3, 2],
                    "max_depth": [1, 10, 0],
                    "n_estimators": [2, 70, 0],
                    "subsample": [0.7, 1, 2],
                }
        elif estimator == 'svm':
            params = {
                "kernel": ["linear", "rbf", "poly"],
                "gamma": [0.1, 10, 1],
                "C": [0.1, 10, 1],
                "degree": [0, 6, 0],
            }
        elif estimator == 'mlp':
            params = {
                "optimizer": ["RMSprop", "adam", "sgd", 'adamax', 'nadam', 'adadelta'],
                "learning_rate": [0.1, 0.3, 2],
                "neurons": [1, 40, 0],
                "hiddenactivation": ['relu', 'sigmoid', 'tanh'],
                "activation": ['relu', 'sigmoid', 'tanh'],
            }
        return params

    @staticmethod
    def get_default_params(estimator, task):
        """Return default parameters for the estimator.

        This method assigns default values for algorithm parameters and
        additional configuration specific to the estimator and task.  It
        allows the user to provide their own hyper‑parameter search space
        while still falling back to sensible defaults for unspecified
        parameters.

        Parameters
        ----------
        estimator : str
            Name of the estimator ('mlp', 'xgboost', 'svm' or 'gbdt').

        task : str
            Task type: 'regression' or 'binary classification'.

        Returns
        -------
        dict
            Default parameter values keyed by parameter name.
        """
        defaultparams = {}
        if estimator == 'xgboost':
            defaultparams.update({'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 40, 'subsample': 0.99})
            if task == 'binary classification':
                defaultparams.update({'objective': 'binary:logistic', 'eval_metric': ["aucpr", "auc"]})
            elif task == 'regression':
                defaultparams.update({'objective': 'reg:tweedie', 'eval_metric': ["rmse"]})
        elif estimator == 'gbdt':
            if task == 'regression':
                defaultparams['objective'] = 'tweedie'
                eval_metric = 'rmse'
            elif task == 'binary classification':
                defaultparams['objective'] = 'binary'
                eval_metric = ['auc']
            defaultparams.update({
                'learning_rate': 0.01,
                'max_depth': 6,
                'n_estimators': 40,
                'subsample': 0.99,
                'boosting_type': 'gbdt',
                'eval_metric': eval_metric,
            })
        elif estimator == 'mlp':
            defaultparams.update({
                'batch_size': 12,
                'epochs': 50,
                'shuffle': True,
                'neurons': 13,
                'hiddenactivation': 'sigmoid',
                'activation': 'sigmoid',
                'learning_rate': 0.01,
                'mode': 'auto',
            })
            if task == 'binary classification':
                defaultparams.update({'optimizer': 'adam', 'metrics': ['binary_accuracy'], 'loss': 'binary_crossentropy'})
            elif task == 'regression':
                defaultparams.update({'optimizer': 'RMSprop', 'metrics': ['mse'], 'loss': 'mse'})
        elif estimator == 'svm':
            defaultparams.update({'kernel': 'rbf', 'C': 5, 'gamma': 5})
        return defaultparams

    @staticmethod
    def read_parameters(params=None, estimator=None, task=None):
        """Parse the user-provided search space into internal structures.

        Parameters
        ----------
        params : dict or None
            User-supplied search space mapping parameter names to lists.

        estimator : str
            Name of the estimator to optimise.

        task : str
            Task type ('regression' or 'binary classification').

        Returns
        -------
        tuple
            Contains (parameters, defaultparams, x_min, x_max, rounding, bounds, dimensions, paramdetails).
        """
        if params is None:
            params = pspso.get_default_search_space(estimator, task)
        x_min, x_max, rounding, parameters = [], [], [], []
        for key in params:
            if all(isinstance(item, str) for item in params[key]):
                # Categorical parameter
                x_min.append(0)
                x_max.append(len(params[key]) - 1)
                parameters.append(key)
                rounding.append(0)
            else:
                thelist = params[key]
                x_min.append(thelist[0])
                x_max.append(thelist[1])
                parameters.append(key)
                rounding.append(thelist[2])
        bounds = (np.asarray(x_min), np.asarray(x_max))
        dimensions = len(x_min)
        defaultparams = pspso.get_default_params(estimator, task)
        return parameters, defaultparams, x_min, x_max, rounding, bounds, dimensions, params

    @staticmethod
    def decode_parameters(particle):
        """Decode an encoded particle into a parameter dictionary.

        This method maps each dimension of a particle back into a
        meaningful parameter value, applying rounding and categorical
        lookups as necessary.  It relies on the global variables
        `parameters`, `defaultparams`, `paramdetails` and `rounding`.

        Parameters
        ----------
        particle : sequence
            Encoded particle position.

        Returns
        -------
        dict
            Decoded parameter dictionary.
        """
        decodeddict = {}
        for d in range(0, len(particle)):
            key = pspso.parameters[d]
            particlevalueatd = particle[d]
            if all(isinstance(item, str) for item in pspso.paramdetails[key]):
                # Categorical: index into the list
                index = int(round(particlevalueatd))
                decodeddict[key] = pspso.paramdetails[key][index]
            else:
                # Numerical: round to specified precision
                decodeddict[key] = round(particlevalueatd, pspso.rounding[pspso.parameters.index(key)])
                if pspso.rounding[pspso.parameters.index(key)] == 0:
                    decodeddict[key] = int(decodeddict[key])
        return decodeddict

    @staticmethod
    def forward_prop_gbdt(particle, task, score, X_train, Y_train, X_val, Y_val):
        """Train a GBDT model for a given particle.

        This function decodes the particle, constructs a LightGBM model
        (`LGBMRegressor` or `LGBMClassifier` depending on the task) and
        fits it on the training data.  It returns the evaluation metric
        for the validation set and the trained model.  Any exceptions
        raised during training are caught and reported with a `None` return.
        """
        # Ensure LightGBM is available
        if lgb is None:
            raise ImportError(
                "lightgbm is required for GBDT optimisation. Install it via `pip install lightgbm`."
            )
        model = None
        eval_set = [(X_val, np.squeeze(Y_val))]
        try:
            decodedparams = pspso.decode_parameters(particle)
            modelparameters = {**pspso.defaultparams, **decodedparams}
            eval_metric = modelparameters['eval_metric']
            del modelparameters['eval_metric']
            if task != 'binary classification':
                model = lgb.LGBMRegressor(**modelparameters)
            else:
                model = lgb.LGBMClassifier(**modelparameters)
            model.fit(
                X_train,
                np.squeeze(Y_train),
                early_stopping_rounds=pspso.early_stopping,
                eval_set=eval_set,
                eval_metric=eval_metric,
                verbose=pspso.verbose,
            )
            return pspso.predict(model, 'gbdt', task, score, X_val, np.squeeze(Y_val)), model
        except Exception as e:
            print('An exception occurred in GBDT training.')
            print(e)
            return None, None

    @staticmethod
    def forward_prop_xgboost(particle, task, score, X_train, Y_train, X_val, Y_val):
        """Train an XGBoost model for a given particle.

        This function decodes the particle, constructs an XGBoost model
        (`XGBRegressor` or `XGBClassifier` depending on the task) and
        fits it on the training data.  It returns the evaluation metric
        for the validation set and the trained model.  Any exceptions
        raised during training are caught and reported with a `None` return.
        """
        # Ensure XGBoost is available
        if xgb is None:
            raise ImportError(
                "xgboost is required for XGBoost optimisation. Install it via `pip install xgboost`."
            )
        model = None
        eval_set = [(X_val, Y_val)]
        try:
            decodedparams = pspso.decode_parameters(particle)
            modelparameters = {**pspso.defaultparams, **decodedparams}
            if task != 'binary classification':
                model = xgb.XGBRegressor(**modelparameters)
            else:
                model = xgb.XGBClassifier(**modelparameters)
            model.fit(
                X_train,
                Y_train,
                eval_set=eval_set,
                verbose=pspso.verbose,
            )
            return pspso.predict(model, 'xgboost', task, score, X_val, Y_val), model
        except Exception as e:
            print('An exception occurred in XGBoost training.')
            print(e)
            return None, None

    @staticmethod
    def forward_prop_svm(particle, task, score, X_train, Y_train, X_val, Y_val):
        """Train an SVM/SVR model for a given particle.

        Depending on the task (regression vs classification) and kernel,
        either an `SVR` or `SVC` is instantiated.  Returns the fitness
        value and the trained model.  Exceptions are caught and printed.
        """
        # Ensure scikit-learn SVM modules are available
        if SVC is None or SVR is None:
            raise ImportError(
                "scikit-learn is required for SVM optimisation. Install it via `pip install scikit-learn`."
            )
        try:
            decodedparams = pspso.decode_parameters(particle)
            modelparameters = {**pspso.defaultparams, **decodedparams}
            if task == 'regression':
                if modelparameters['kernel'] != 'poly':
                    model = SVR(
                        kernel=modelparameters['kernel'],
                        C=modelparameters['C'],
                        gamma=modelparameters['gamma'],
                    ).fit(X_train, np.squeeze(Y_train))
                else:
                    model = SVR(
                        kernel=modelparameters['kernel'],
                        C=modelparameters['C'],
                        gamma=modelparameters['gamma'],
                        degree=modelparameters['degree'],
                    ).fit(X_train, np.squeeze(Y_train))
            elif task == 'binary classification':
                if modelparameters['kernel'] != 'poly':
                    model = SVC(
                        kernel=modelparameters['kernel'],
                        C=modelparameters['C'],
                        gamma=modelparameters['gamma'],
                        probability=True,
                    ).fit(X_train, np.squeeze(Y_train))
                else:
                    model = SVC(
                        kernel=modelparameters['kernel'],
                        C=modelparameters['C'],
                        gamma=modelparameters['gamma'],
                        degree=modelparameters['degree'],
                        probability=True,
                    ).fit(X_train, np.squeeze(Y_train))
            return pspso.predict(model, 'svm', task, score, X_val, Y_val), model
        except Exception as e:
            print(e)
            print('An exception occurred in SVM training.')
            return None, None

    @staticmethod
    def forward_prop_mlp(particle, task, score, X_train, Y_train, X_val, Y_val):
        """Train an MLP model for a given particle.

        Constructs a simple fully connected neural network using
        `tensorflow.keras`.  A two‑layer architecture is used with a single
        hidden layer; other architectures are intentionally out of scope
        for this package.  Returns the fitness value, the trained model
        and the training history.  Exceptions are caught and reported.
        """
        # Ensure TensorFlow/Keras is available
        if Sequential is None or Dense is None or EarlyStopping is None:
            raise ImportError(
                "TensorFlow with Keras is required for MLP optimisation. Install it via `pip install tensorflow`."
            )
        try:
            decodedparams = pspso.decode_parameters(particle)
            modelparameters = {**pspso.defaultparams, **decodedparams}
            model = Sequential()
            model.add(
                Dense(
                    int(modelparameters['neurons']),
                    input_shape=(X_train.shape[1],),
                    activation=modelparameters['hiddenactivation'],
                )
            )
            model.add(Dense(1, activation=modelparameters['activation']))
            model.compile(
                loss=modelparameters['loss'],
                optimizer=modelparameters['optimizer'],
                metrics=modelparameters['metrics'],
            )
            model.optimizer.learning_rate = modelparameters['learning_rate']
            es = EarlyStopping(
                monitor='val_loss',
                mode=modelparameters['mode'],
                verbose=pspso.verbose,
                patience=pspso.early_stopping,
            )
            history = model.fit(
                X_train,
                Y_train,
                batch_size=modelparameters['batch_size'],
                epochs=modelparameters['epochs'],
                shuffle=modelparameters['shuffle'],
                validation_data=(X_val, Y_val),
                callbacks=[es],
                verbose=pspso.verbose,
            )
            return pspso.predict(model, 'mlp', task, score, X_val, Y_val), model, history
        except Exception as e:
            print("An exception occurred in MLP training.")
            print(e)
            return None, None

    @staticmethod
    def f(q, estimator, task, score, X_train, Y_train, X_val, Y_val):
        """Apply forward propagation across the swarm.

        Given a swarm `q` of particle positions, evaluate the fitness
        function for each particle by forwarding to the appropriate
        algorithm-specific `forward_prop_` function.  Handles special
        bookkeeping for the MLP case to store the best model across
        iterations.  Returns an array of cost values corresponding to the
        particles in `q`.
        """
        n_particles = q.shape[0]
        if estimator == 'xgboost':
            e = [pspso.forward_prop_xgboost(q[i], task, score, X_train, Y_train, X_val, Y_val) for i in range(n_particles)]
            j = [e[i][0] for i in range(n_particles)]
        elif estimator == 'gbdt':
            e = [pspso.forward_prop_gbdt(q[i], task, score, X_train, Y_train, X_val, Y_val) for i in range(n_particles)]
            j = [e[i][0] for i in range(n_particles)]
        elif estimator == 'svm':
            e = [pspso.forward_prop_svm(q[i], task, score, X_train, Y_train, X_val, Y_val) for i in range(n_particles)]
            j = [e[i][0] for i in range(n_particles)]
        elif estimator == 'mlp':
            e = [pspso.forward_prop_mlp(q[i], task, score, X_train, Y_train, X_val, Y_val) for i in range(n_particles)]
            j = [e[i][0] for i in range(n_particles)]
            if pspso.best_particle_position_ann is not None:
                # Adjust cost history to preserve the best model across iterations
                for i in range(n_particles):
                    if pspso.decode_parameters(q[i]) == pspso.decode_parameters(pspso.best_particle_position_ann):
                        if j[i] > pspso.best_paricle_cost_ann:
                            j[i] = pspso.best_paricle_cost_ann
                    if min(j) <= pspso.best_paricle_cost_ann:
                        min_loss_index = j.index(min(j))
                        pspso.best_paricle_cost_ann = min(j)
                        pspso.best_model_ann = e[min_loss_index][1]
                        pspso.best_history_ann = e[min_loss_index][2]
                        pspso.best_particle_position_ann = q[min_loss_index]
            else:
                min_loss_index = j.index(min(j))
                pspso.best_paricle_cost_ann = min(j)
                pspso.best_model_ann = e[min_loss_index][1]
                pspso.best_history_ann = e[min_loss_index][2]
                pspso.best_particle_position_ann = q[min_loss_index]
        return np.array(j)

    @staticmethod
    def rebuildmodel(estimator, pos, task, score, X_train, Y_train, X_val, Y_val):
        """Rebuild the best model based on the global best position.

        For algorithms other than MLP, simply forward the position to the
        corresponding training function.  For MLP, return the stored best
        model and cost from previous runs.
        """
        if estimator == 'xgboost':
            met, model = pspso.forward_prop_xgboost(pos, task, score, X_train, Y_train, X_val, Y_val)
        elif estimator == 'gbdt':
            met, model = pspso.forward_prop_gbdt(pos, task, score, X_train, Y_train, X_val, Y_val)
        elif estimator == 'svm':
            met, model = pspso.forward_prop_svm(pos, task, score, X_train, Y_train, X_val, Y_val)
        elif estimator == 'mlp':
            return pspso.best_paricle_cost_ann, pspso.best_model_ann
        return met, model

    def fitpspso(
        self,
        X_train=None,
        Y_train=None,
        X_val=None,
        Y_val=None,
        psotype='global',
        number_of_particles=5,
        number_of_iterations=10,
        options={'c1': 1.49618, 'c2': 1.49618, 'w': 0.7298},
    ):
        """Run a PSO hyper‑parameter search.

        Parameters
        ----------
        X_train, Y_train, X_val, Y_val : array-like
            Training and validation splits.  See class docstring.

        psotype : str, optional
            One of 'global' or 'local'.  Determines whether a global or
            local PSO variant is used.

        number_of_particles : int, optional
            Number of particles in the swarm.

        number_of_iterations : int, optional
            Number of PSO iterations.

        options : dict, optional
            Dictionary of PSO hyper‑parameters (`c1`, `c2` and `w`).

        Returns
        -------
        tuple
            Encoded best position, cost, duration, trained model and the
            optimizer instance.
        """
        # Ensure PySwarms is available before starting optimisation
        if ps is None:
            raise ImportError(
                "pyswarms is required for PSO optimisation. Install it via `pip install pyswarms`."
            )
        print("Running PSO Search .....")
        self.selectiontype = "PSO"
        self.number_of_particles = number_of_particles
        self.number_of_iterations = number_of_iterations
        self.psotype = psotype
        self.options = options
        self.number_of_attempts = self.number_of_iterations * self.number_of_particles
        self.totalnbofcombinations = len(self.calculatecombinations())
        pspso.best_paricle_cost_ann = None
        pspso.best_model_ann = None
        pspso.best_history_ann = None
        pspso.best_particle_position_ann = None
        kwargs = {
            "estimator": self.estimator,
            "task": self.task,
            "score": self.score,
            "X_train": X_train,
            "Y_train": Y_train,
            "X_val": X_val,
            "Y_val": Y_val,
        }
        if psotype == 'global':
            self.optimizer = ps.single.GlobalBestPSO(
                n_particles=self.number_of_particles,
                dimensions=self.dimensions,
                options=self.options,
                bounds=self.bounds,
            )
        elif psotype == 'local':
            self.optimizer = ps.single.LocalBestPSO(
                n_particles=self.number_of_particles,
                dimensions=self.dimensions,
                options=self.options,
                bounds=self.bounds,
            )
        start = time.time()
        self.cost, self.pos = self.optimizer.optimize(pspso.f, iters=self.number_of_iterations, **kwargs)
        end = time.time()
        self.duration = end - start
        self.met, self.model = pspso.rebuildmodel(
            self.estimator,
            self.pos,
            self.task,
            self.score,
            X_train,
            Y_train,
            X_val,
            Y_val,
        )
        if self.estimator == 'mlp':
            self.history = pspso.best_history_ann
        self.miniopt = self.save_optimizer_details()
        return self.pos, self.cost, self.duration, self.model, self.optimizer

    def fitpsgrid(self, X_train=None, Y_train=None, X_val=None, Y_val=None):
        """Perform a grid search over all combinations in the search space."""
        print("Running Grid Search .....")
        self.selectiontype = "Grid"
        self.results = []
        self.model = None
        self.pos = None
        self.cost = None
        self.combinations = self.calculatecombinations()
        self.totalnbofcombinations = len(self.combinations)
        self.number_of_attempts = self.totalnbofcombinations
        start = time.time()
        for comb in self.combinations:
            if self.estimator == 'xgboost':
                met, mo = pspso.forward_prop_xgboost(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            elif self.estimator == 'gbdt':
                met, mo = pspso.forward_prop_gbdt(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            elif self.estimator == 'svm':
                met, mo = pspso.forward_prop_svm(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            elif self.estimator == 'mlp':
                met, mo, hist = pspso.forward_prop_mlp(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            self.results.append(met)
            if self.cost is None:
                self.cost = met
                self.pos = comb
                self.model = mo
            else:
                if met < self.cost:
                    self.cost = met
                    self.pos = comb
                    self.model = mo
                    if self.estimator == 'mlp':
                        self.history = hist
        end = time.time()
        self.duration = end - start
        return self.pos, self.cost, self.duration, self.model, self.combinations, self.results

    def fitpsrandom(self, X_train=None, Y_train=None, X_val=None, Y_val=None, number_of_attempts=20):
        """Perform a random search over a subset of the search space."""
        print("Running Random Search .....")
        self.number_of_attempts = number_of_attempts
        self.selectiontype = "Random"
        self.combinations = self.calculatecombinations()
        self.totalnbofcombinations = len(self.combinations)
        start = time.time()
        self.results = []
        self.model = None
        self.pos = None
        self.cost = None
        for _ in range(number_of_attempts):
            a = random.randint(0, self.totalnbofcombinations - 1)
            comb = self.combinations[a]
            if self.estimator == 'xgboost':
                met, mo = pspso.forward_prop_xgboost(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            elif self.estimator == 'gbdt':
                met, mo = pspso.forward_prop_gbdt(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            elif self.estimator == 'svm':
                met, mo = pspso.forward_prop_svm(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            elif self.estimator == 'mlp':
                met, mo, hist = pspso.forward_prop_mlp(comb, self.task, self.score, X_train, Y_train, X_val, Y_val)
            self.results.append(met)
            if self.cost is None:
                self.cost = met
                self.pos = comb
                self.model = mo
            else:
                if met < self.cost:
                    self.cost = met
                    self.pos = comb
                    self.model = mo
                    if self.estimator == 'mlp':
                        self.history = hist
        end = time.time()
        self.duration = end - start
        return self.pos, self.cost, self.duration, self.model, self.combinations, self.results

    def print_results(self):
        """Pretty-print the results of the search."""
        print("Estimator: " + self.estimator)
        print("Task: " + self.task)
        print("Selection type: " + str(self.selectiontype))
        print("Number of attempts:" + str(self.number_of_attempts))
        print("Total number of combinations: " + str(self.totalnbofcombinations))
        print("Parameters:")
        print(pspso.decode_parameters(self.pos))
        print("Global best position: " + str(self.pos))
        print("Global best cost: " + str(round(self.cost, 4)))
        print("Time taken to find the set of parameters: " + str(self.duration))
        if self.selectiontype == "PSO":
            print("Number of particles: " + str(self.number_of_particles))
            print("Number of iterations: " + str(self.number_of_iterations))

    def calculatecombinations(self):
        """Generate all possible combinations in the search space."""
        index = 0
        thedict = {}
        for i, j in zip(self.x_min, self.x_max):
            # Create a range respecting the rounding precision
            a = np.arange(i, j + 0.000001, 10 ** (-1 * pspso.rounding[index]))
            a = np.round(a, pspso.rounding[index])
            thedict[pspso.parameters[index]] = a
            index = index + 1
        combinations = it.product(*(thedict[Name] for Name in thedict))
        combinations = list(combinations)
        combinations = [list(row) for row in combinations]
        return combinations

    @staticmethod
    def predict(model, estimator, task, score, X_val, Y_val):
        """Compute the fitness of a model on validation data."""
        if score == 'rmse':
            preds_val = model.predict(X_val)
            met = np.sqrt(mean_squared_error(Y_val, preds_val))
            return met
        if task == 'binary classification' and estimator == 'mlp':
            preds_val = model.predict(X_val)
            if score == 'acc':
                met = accuracy_score(Y_val, np.round(preds_val))
                return 1 - met
            elif score == 'auc':
                fpr, tpr, _ = roc_curve(Y_val, preds_val)
                met = auc(fpr, tpr)
                return 1 - met
        elif task == 'binary classification' and (estimator == 'xgboost' or estimator == 'svm' or estimator == 'gbdt'):
            if score == 'acc':
                preds_val = model.predict(X_val)
                met = accuracy_score(Y_val, preds_val)
                return 1 - met
            elif score == 'auc':
                preds_val = model.predict_proba(X_val)
                fpr, tpr, _ = roc_curve(Y_val, preds_val[:, 1])
                met = auc(fpr, tpr)
                return 1 - met

    def save_optimizer_details(self):
        """Serialize PSO optimiser details for later inspection."""
        opt = {}
        opt['pos_history'] = self.optimizer.pos_history
        opt['cost_history'] = self.optimizer.cost_history
        opt['bounds'] = self.optimizer.bounds
        opt['init_pos'] = self.optimizer.init_pos
        opt['swarm_size'] = self.optimizer.swarm_size
        opt['options'] = self.optimizer.options
        opt['name'] = self.optimizer.name
        opt['n_particles'] = self.optimizer.n_particles
        opt['cost_history'] = self.optimizer.cost_history
        return opt