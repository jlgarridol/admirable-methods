from sklearn.utils import check_random_state, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MetaEstimatorMixin, is_classifier
from abc import ABCMeta, abstractmethod
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from joblib import effective_n_jobs, Parallel, delayed



class RFWTree(BaseEstimator):
    __metaclass__ = ABCMeta
    """Base class for Random Features Weights trees.
    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, exponent, criterion, splitter,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features, random_state,
                 max_leaf_nodes, min_impurity_decrease, ccp_alpha):
        """
        Jesús Maudes, Juan J. Rodríguez, César García-Osorio, Nicolás García-Pedrajas,
        Random feature weights for decision tree ensemble construction,
        Information Fusion, Volume 13, Issue 1, 2012,
        Pages 20-30, ISSN 1566-2535,
        https://doi.org/10.1016/j.inffus.2010.11.004.
        """
        self.exponent = exponent
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Build a RFW tree classifier or regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Classification: The target values (class labels) as integers or strings.
            Regression: The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        self : RFWTree
            Fitted estimator.
        """
        rs = check_random_state(self.random_state)
        features = X.shape[1]
        weights = rs.uniform(0, 1, features)**self.exponent
        self.tree_ = self.create_tree()
        try:
            self.tree_.fit(X, y, sample_weight=sample_weight,
                           check_input=check_input, feature_weight=weights)
        except TypeError as e:
            if "unexpected keyword argument 'feature_weight'" in str(e):
                raise TypeError(
                    "The vanilla sklearn not support feature_weight, use version sklearn-ubu (https://github.com/jlgarridol/sklearn-ubu/releases).")
            else:
                raise e

        return self

    @abstractmethod
    def create_tree(self):
        """Generate a tree object.
        Returns
        -------
        tree : DecisionTreeClassifier or DecisionTreeRegressor
        """
        pass

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        return self.tree_.predict(X, check_input=check_input)


class RFWTreeClassifier(RFWTree, ClassifierMixin):

    def __init__(self, exponent=1, criterion='entropy', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 class_weight=None, ccp_alpha=0.0):
        """A Random Forest Weight tree classifier.

        Parameters
        ----------
        exponent : int, default=1
            Exponent of the random weights.
         criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.
        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.
        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.
        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:
            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.
            .. versionchanged:: 0.18
            Added float values for fractions.
        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.
            .. versionchanged:: 0.18
            Added float values for fractions.
        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.
        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:
                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a fraction and
                `max(1, int(max_features * n_features_in_))` features are considered at
                each split.
                - If "auto", then `max_features=sqrt(n_features)`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.
                .. deprecated:: 1.1
                    The `"auto"` option was deprecated in 1.1 and will be removed
                    in 1.3.
            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.
        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.
            The weighted impurity decrease equation is the following::
                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)
            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.
            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.
            .. versionadded:: 0.19
        class_weight : dict, list of dict or "balanced", default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If None, all classes are supposed to have weight one. For
            multi-output problems, a list of dicts can be provided in the same
            order as the columns of y.
            Note that for multioutput (including multilabel) weights should be
            defined for each class of every column in its own dict. For example,
            for four-class multilabel classification weights should be
            [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
            [{1:1}, {2:5}, {3:1}, {4:1}].
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``
            For multi-output, the weights of each column of y will be multiplied.
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
        ccp_alpha : non-negative float, default=0.0
            Complexity parameter used for Minimal Cost-Complexity Pruning. The
            subtree with the largest cost complexity that is smaller than
            ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
            :ref:`minimal_cost_complexity_pruning` for details.
        """
        super().__init__(exponent, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         ccp_alpha)
        self.class_weight = class_weight

    def create_tree(self):
        return DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter,
                                      max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf,
                                      min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                      max_features=self.max_features, random_state=self.random_state,
                                      max_leaf_nodes=self.max_leaf_nodes,
                                      min_impurity_decrease=self.min_impurity_decrease,
                                      class_weight=self.class_weight, ccp_alpha=self.ccp_alpha)

    def predict_proba(self, X):
        return self.tree_.predict_proba(X)


class RFWTreeRegressor(RFWTree, RegressorMixin):

    def __init__(self, exponent=1, criterion='squared_error', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None,
                 random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, ccp_alpha=0.0):
        super().__init__(exponent, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         ccp_alpha)

    def create_tree(self):
        return DecisionTreeRegressor(criterion=self.criterion, splitter=self.splitter,
                                     max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                     min_samples_leaf=self.min_samples_leaf,
                                     min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                     max_features=self.max_features, random_state=self.random_state,
                                     max_leaf_nodes=self.max_leaf_nodes,
                                     min_impurity_decrease=self.min_impurity_decrease,
                                     ccp_alpha=self.ccp_alpha)


class RFW(BaseEstimator, MetaEstimatorMixin):
    __metaclass__ = ABCMeta

    def __init__(self, n_estimators, exponent, criterion, splitter,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features, random_state,
                 max_leaf_nodes, min_impurity_decrease, ccp_alpha, n_jobs):
        self.n_estimators = n_estimators
        self.exponent = exponent
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.n_jobs = n_jobs

    @abstractmethod
    def define_tree(self, random_state):
        pass

    def __create_new_tree(self, random_state):
        rs = random_state.randint(0, 2**32)
        return self.define_tree(rs)

    def _fit_estimator(self, tree, X, y, sample_weight, check_input):
        tree.fit(X, y, sample_weight, check_input)
        return tree

    def fit(self, X, y, sample_weight=None, check_input=True):
        X, y = check_X_y(X, y)
        if is_classifier(self):
            self.classes_ = np.unique(y)

        n_jobs = min(effective_n_jobs(self.n_jobs), self.n_estimators)
        random_state = check_random_state(self.random_state)

        self.trees_ = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_estimator)(
                self.__create_new_tree(random_state),
                X,
                y,
                sample_weight,
                check_input
            )
            for i in range(self.n_estimators)
        )
        return self

    @abstractmethod
    def predict(self, X):
        pass


class RFWClassifier(RFW, ClassifierMixin):

    def __init__(self, n_estimators=10, exponent=1, criterion='entropy', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 class_weight=None, ccp_alpha=0.0, n_jobs=None):
        super().__init__(n_estimators, exponent, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         ccp_alpha, n_jobs)
        self.class_weight = class_weight

    def define_tree(self, random_state):
        return RFWTreeClassifier(exponent=self.exponent, criterion=self.criterion, splitter=self.splitter,
                                 max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                 max_features=self.max_features, random_state=random_state,
                                 max_leaf_nodes=self.max_leaf_nodes,
                                 min_impurity_decrease=self.min_impurity_decrease,
                                 class_weight=self.class_weight, ccp_alpha=self.ccp_alpha)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.trees_:
            predictions += tree.predict_proba(X)
        return self.classes_[np.argmax(predictions, axis=1)]

    def predict_proba(self, X):
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.trees_:
            predictions += tree.predict_proba(X)
        return predictions / len(self.trees_)


class RFWRegressor(RFW, RegressorMixin):

    def __init__(self, n_estimators=10, exponent=1, criterion='squared_error', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, n_jobs=None):
        super().__init__(n_estimators, exponent, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         ccp_alpha, n_jobs)

    def define_tree(self, random_state):
        return RFWTreeRegressor(exponent=self.exponent, criterion=self.criterion, splitter=self.splitter,
                                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                max_features=self.max_features, random_state=random_state,
                                max_leaf_nodes=self.max_leaf_nodes,
                                min_impurity_decrease=self.min_impurity_decrease,
                                ccp_alpha=self.ccp_alpha)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.trees_:
            predictions += tree.predict(X)
        return predictions / len(self.trees_)
