import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, check_X_y
from sklearn.base import clone as skclone


class DisturbingNeighborsClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator, n_neighbors=1, disturbing_features=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_neighbors = n_neighbors
        self.disturbing_features = disturbing_features
        self.random_state = random_state

    def __increase_features(self, X):

        nearest_class = self.nn1_.predict(X)
        nearest_neighbor = self.one_hot_.transform(self.nn1_.kneighbors(X, return_distance=False)).astype(bool)

        return np.concatenate((X, nearest_class.reshape(-1, 1), nearest_neighbor), axis=1)

    def fit(self, X, y, sample_weight=None):
        
        self.base_estimator = skclone(self.base_estimator)
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        rs = check_random_state(self.random_state)

        # Select c random features
        instances = X.shape[0]
        c = self.disturbing_features
        self.instances_ = rs.choice(instances, c, False)
        self.one_hot_ = OneHotEncoder(sparse=False)
        self.one_hot_.fit(np.arange(c).reshape(-1, 1))

        # Create 1NN classifier
        self.nn1_ = KNeighborsClassifier(n_neighbors=1).fit(np.unique(X, axis=0)[self.instances_, :], y[self.instances_])
        X_large = self.__increase_features(X)

        self.base_estimator.fit(X_large, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        X_large = self.__increase_features(X)

        return self.base_estimator.predict_proba(X_large)

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
