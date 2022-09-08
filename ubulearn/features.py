from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MetaEstimatorMixin, is_classifier
from abc import ABCMeta, abstractmethod
import numpy as np


class RFWTree(BaseEstimator):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, exponent, criterion, splitter,
                 max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features, random_state,
                 max_leaf_nodes, min_impurity_decrease, ccp_alpha):
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
        rs = check_random_state(self.random_state)
        features = X.shape[1]
        weights = rs.uniform(0, 1, features)**self.exponent
        self.tree_ = self.create_tree()
        self.tree_.fit(X, y, sample_weight=sample_weight, check_input=check_input, feature_weight=weights)

        return self

    @abstractmethod
    def create_tree(self):
        pass

    def predict(self, X):
        return self.tree_.predict(X)


class RFWTreeClassifier(RFWTree, ClassifierMixin):

    def __init__(self, exponent=1, criterion='entropy', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 class_weight=None, ccp_alpha=0.0):
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
                 max_leaf_nodes, min_impurity_decrease, ccp_alpha):
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

    @abstractmethod
    def define_tree(self, random_state):
        pass

    def __create_new_tree(self, random_state):
        rs = random_state.randint(0, 2**32)
        return self.define_tree(rs)

    def fit(self, X, y, sample_weight=None, check_input=True):
        X, y = check_X_y(X, y)
        if is_classifier(self):
            self.classes_ = np.unique(y)
        self.trees_ = []
        random_state = check_random_state(self.random_state)
        for i in range(self.n_estimators):
            tree = self.__create_new_tree(random_state)
            tree.fit(X, y, sample_weight, check_input)
            self.trees_.append(tree)

        return self

    @abstractmethod
    def predict(self, X):
        pass


class RFWClassifier(RFW, ClassifierMixin):

    def __init__(self, n_estimators=10, exponent=1, criterion='entropy', splitter='best',
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 class_weight=None, ccp_alpha=0.0):
        super().__init__(n_estimators, exponent, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         ccp_alpha)
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
                 max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0):
        super().__init__(n_estimators, exponent, criterion, splitter, max_depth, min_samples_split,
                         min_samples_leaf, min_weight_fraction_leaf, max_features,
                         random_state, max_leaf_nodes, min_impurity_decrease,
                         ccp_alpha)

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