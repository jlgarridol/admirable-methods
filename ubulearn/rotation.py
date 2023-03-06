"""Rotation Forest
Implementation based on:
https://github.com/alan-turing-institute/sktime/blob/cc91ba9591aa88cba3874365782951745cd5ad6d/sktime/classification/sklearn/_rotation_forest.py
"""

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, MetaEstimatorMixin, is_classifier
from abc import abstractmethod, ABCMeta
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import clone as skclone
from sklearn.utils import check_X_y, check_random_state, check_array
from sklearn.decomposition import PCA


class RotationTransformer(TransformerMixin):

    def __init__(self, min_group=3, max_group=3, remove_proportion=0.5, n_jobs=None, random_state=None):
        """
        Tranformer that rotates the features of a dataset.

        Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006).
        Rotation forest: A new classifier ensemble method.
        IEEE transactions on pattern analysis and machine intelligence,
        28(10), 1619-1630.

        Parameters
        ----------
        min_group : int, optional
            Minimum size of a group of attributes, by default 3
        max_group : int, optional
            Maximum size of a group of attributes, by default 3
        remove_proportion : float, optional
            Proportion of instances to be removed, by default 0.5
        n_jobs : int, optional
            The number of jobs to run in parallel for both `fit` and `predict`.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """

        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fit the transformer according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,), default=None
            The target values (class labels) if supervised.
            If None, unsupervised learning is assumed.

        Returns
        -------
        self : RotationTranformer
            Returns self.
        """        
        X = check_array(X)
        self.n_features_ = X.shape[1]
        if y is not None:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            X_cls_split = [X[np.where(y == i)] for i in self.classes_]
        else:
            X_cls_split = None
        self.n_jobs_ = effective_n_jobs(self.n_jobs)
        self.random_state_ = check_random_state(self.random_state)

        self.groups_ = self._generate_groups(self.random_state_)
        self.pcas_ = self._generate_pcas(
            X_cls_split, self.groups_, self.random_state_)

        return self

    def transform(self, X):
        """
        Transform the data according to the fitted transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_t : array-like of shape (n_samples, n_features)
            The transformed samples.
        """
        X_t = np.concatenate(
            [self.pcas_[i].transform(X[:, group]) for i, group in enumerate(self.groups_)], axis=1
        )
        return X_t

    def _generate_pcas(self, X_cls_split, groups, rng):
        pcas = []
        for group in groups:
            classes = rng.choice(
                range(self.n_classes_),
                size=rng.randint(1, self.n_classes_ + 1),
                replace=False,
            )

            X_t = np.zeros((0, len(group)))
            if X_cls_split is not None:
                # randomly add the classes with the randomly selected attributes.
                for cls_idx in classes:
                    c = X_cls_split[cls_idx]
                    X_t = np.concatenate((X_t, c[:, group]), axis=0)

            sample_ind = rng.choice(
                X_t.shape[0],
                max(1, int(X_t.shape[0] * self.remove_proportion)),
                replace=False,
            )
            X_t = X_t[sample_ind]

            # try to fit the PCA if it fails, remake it, and add 10 random data
            # instances.
            while True:
                # ignore err state on PCA because we account if it fails.
                with np.errstate(divide="ignore", invalid="ignore"):
                    # differences between os occasionally. seems to happen when there
                    # are low amounts of cases in the fit
                    pca = PCA(random_state=rng.randint(1, 255)).fit(X_t)

                if not np.isnan(pca.explained_variance_ratio_).all():
                    break
                X_t = np.concatenate(
                    (X_t, rng.random_sample((10, X_t.shape[1]))), axis=0
                )

            pcas.append(pca)
        return pcas

    def _generate_groups(self, rng):
        """Generate random groups of subspaces. The size of each group is randomly selected between
        min_group and max_group. If the number of features is not divisible by the size of the group,
        the last group will have repeated random attributes added to it.

        Parameters
        ----------
        rng : RandomState
            Random state.

        Returns
        -------
        list
            List of groups of subspaces.
        """        
        # FROM: https://github.com/alan-turing-institute/sktime/blob/cc91ba9591aa88cba3874365782951745cd5ad6d/sktime/classification/sklearn/_rotation_forest.py#L488
        permutation = rng.permutation((np.arange(0, self.n_features_)))

        # select the size of each group.
        group_size_count = np.zeros(self.max_group - self.min_group + 1)
        n_attributes = 0
        n_groups = 0
        while n_attributes < self.n_features_:
            n = rng.randint(group_size_count.shape[0])
            group_size_count[n] += 1
            n_attributes += self.min_group + n
            n_groups += 1

        groups = []
        current_attribute = 0
        current_size = 0
        for i in range(0, n_groups):
            while group_size_count[current_size] == 0:
                current_size += 1
            group_size_count[current_size] -= 1

            n = self.min_group + current_size
            groups.append(np.zeros(n, dtype=int))
            for k in range(0, n):
                if current_attribute < permutation.shape[0]:
                    groups[i][k] = permutation[current_attribute]
                else:
                    groups[i][k] = permutation[rng.randint(
                        permutation.shape[0])]
                current_attribute += 1

        return groups


class RotationTree(BaseEstimator, MetaEstimatorMixin):
    ___metaclass__ = ABCMeta

    def __init__(self, base_estimator, min_group=3, max_group=3, remove_proportion=0.5, n_jobs=None, random_state=None):
        """
        This is a rotation tree. It is a tree that uses a rotation forest to generate the ensemble. 
        Do not use this class directly, use the derived classes instead.
        
        Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006).
        Rotation forest: A new classifier ensemble method.
        IEEE transactions on pattern analysis and machine intelligence,
        28(10), 1619-1630.

        Parameters
        ----------
        base_estimator : BaseEstimator
            Estimator to use for the rotation tree.
        min_group : int, optional
            Minimum size of a group of attributes, by default 3
        max_group : int, optional
            Maximum size of a group of attributes, by default 3
        remove_proportion : float, optional
            Proportion of instances to be removed, by default 0.5
        n_jobs : int, optional
            The number of jobs to run in parallel for both `fit` and `predict`.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        self.base_estimator = base_estimator
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, **kwards):
        """
        Fit the rotation tree.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : RotationTree
            Fitted estimator.
        """        
        X, y = check_X_y(X, y)
        self.n_features_ = X.shape[1]
        self.random_state_ = check_random_state(self.random_state)
        self.rotation_ = RotationTransformer(
            min_group=self.min_group,
            max_group=self.max_group,
            n_jobs=self.n_jobs,
            remove_proportion=self.remove_proportion,
            random_state=self.random_state_)

        X_transformed = self.rotation_.fit_transform(X, y)

        self.estimator_ = skclone(self.base_estimator).fit(
            X_transformed, y, **kwards)

        return self

    @abstractmethod
    def predict(self, X):
        pass


class RotationTreeClassifier(RotationTree, ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(criterion="entropy"), min_group=3, max_group=3, remove_proportion=0.5, random_state=None):
        super(RotationTreeClassifier, self).__init__(
            base_estimator=base_estimator,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            random_state=random_state,
        )
        self._estimator_type = "classifier"

    def fit(self, X, y, **kwards):
        super(RotationTreeClassifier, self).fit(X, y, **kwards)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """        
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the classes corresponds to that in the attribute `classes_`.
        """
        X = check_array(X)
        X_transformed = self.rotation_.transform(X)
        return self.estimator_.predict_proba(X_transformed)


class RotationTreeRegressor(RotationTree, RegressorMixin):

    def __init__(self, base_estimator=DecisionTreeRegressor(criterion="squared_error"), min_group=3, max_group=3, remove_proportion=0.5, random_state=None):
        super(RotationTreeRegressor, self).__init__(
            base_estimator=base_estimator,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            random_state=random_state,
        )

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = check_array(X)
        X_transformed = self.rotation_.transform(X)
        return self.estimator_.predict(X_transformed)


class RotationForest(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, base_estimator, n_estimators=100, min_group=3, max_group=3, remove_proportion=0.5, random_state=None, n_jobs=None):
        """
        This is a rotation forest.
        Do not use this class directly, use the derived classes instead.
        
        Rodriguez, J. J., Kuncheva, L. I., & Alonso, C. J. (2006).
        Rotation forest: A new classifier ensemble method.
        IEEE transactions on pattern analysis and machine intelligence,
        28(10), 1619-1630.

        Parameters
        ----------        
        base_estimator : BaseEstimator
            Estimator to use for the rotation tree.
        n_estimators : int, optional
            number of trees, by default 100
        min_group : int, optional
            Minimum size of a group of attributes, by default 3
        max_group : int, optional
            Maximum size of a group of attributes, by default 3
        remove_proportion : float, optional
            Proportion of instances to be removed, by default 0.5
        n_jobs : int, optional
            The number of jobs to run in parallel for both `fit` and `predict`.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.min_group = min_group
        self.max_group = max_group
        self.remove_proportion = remove_proportion
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwards):
        rs = check_random_state(self.random_state)
        X, y = check_X_y(X, y)
        if is_classifier(self):
            self.base_ = RotationTreeClassifier(
                self.base_estimator,
                self.min_group,
                self.max_group,
                self.remove_proportion
            )
        else:
            self.base_ = RotationTreeRegressor(
                self.base_estimator,
                self.min_group,
                self.max_group,
                self.remove_proportion
            )

        # Remove useless attributes
        self._useful_atts = ~np.all(X[1:] == X[:-1], axis=0)
        X = X[:, self._useful_atts]
        # Normalize attributes
        self._min = X.min(axis=0)
        self._ptp = X.max(axis=0) - self._min
        X = (X - self._min) / self._ptp

        self.n_jobs_ = min(effective_n_jobs(self.n_jobs), self.n_estimators)
        self.trees_ = Parallel(n_jobs=self.n_jobs_)(
            delayed(self._fit_estimator)(
                skclone(self.base_),
                X,
                y,
                rs.randint(np.iinfo(np.int32).max),
                **kwards
            )
            for i in range(self.n_estimators)
        )
        return self

    def _fit_estimator(self, estimator, X, y, random_state, **kwards):
        random_state = check_random_state(random_state)

        to_set = {}
        for key in sorted(estimator.get_params(deep=True)):
            if key == "random_state" or key.endswith("__random_state"):
                to_set[key] = random_state.randint(np.iinfo(np.int32).max)

        if to_set:
            estimator.set_params(**to_set)

        return estimator.fit(X, y, **kwards)

    def predict(self, X):
        if is_classifier(self):
            rng = check_random_state(self.random_state)
            return np.array(
                [
                    self.classes_[
                        int(rng.choice(np.flatnonzero(prob == prob.max())))]
                    for prob in self.predict_proba(X)
                ]
            )
        else:
            return self.predict_proba(X)

    def predict_proba(self, X):
        X = check_array(X)
        X = X[:, self._useful_atts]
        X = (X - self._min) / self._ptp

        y_probas = Parallel(n_jobs=self.n_jobs_)(
            delayed(self._predict_proba_for_estimator)(
                X,
                self.trees_[i],
            )
            for i in range(self.n_estimators)
        )

        # y_probas = [self._predict_proba_for_estimator(X, self.trees_[i]) for i in range(self.n_estimators) ]

        if is_classifier(self):
            output = np.sum(y_probas, axis=0) / (
                np.ones(len(self.classes_)) * self.n_estimators
            )
        else:
            output = np.mean(y_probas, axis=0)
        return output

    def _predict_proba_for_estimator(self, X, estimator):

        if is_classifier(self):
            probas = estimator.predict_proba(X)
            if probas.shape[1] != len(self.classes_):
                new_probas = np.zeros((probas.shape[0], len(self.classes_)))
                for i, cls in enumerate(estimator.classes_):
                    cls_idx = self._class_dictionary[cls]
                    new_probas[:, cls_idx] = probas[:, i]
                probas = new_probas
        else:
            probas = estimator.predict(X)

        return probas


class RotationForestClassifier(RotationForest, ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(criterion="entropy"), n_estimators=100, min_group=3, max_group=3, remove_proportion=0.5, random_state=None, n_jobs=None):
        super(RotationForestClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._estimator_type = "classifier"

    def fit(self, X, y, **kwards):
        super(RotationForestClassifier, self).fit(X, y, **kwards)
        self.classes_ = np.unique(y)
        return self


class RotationForestRegressor(RotationForest, RegressorMixin):

    def __init__(self, base_estimator=DecisionTreeRegressor(criterion="squared_error"), n_estimators=100, min_group=3, max_group=3, remove_proportion=0.5, random_state=None, n_jobs=None):
        super(RotationForestRegressor, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            min_group=min_group,
            max_group=max_group,
            remove_proportion=remove_proportion,
            random_state=random_state,
            n_jobs=n_jobs,
        )
