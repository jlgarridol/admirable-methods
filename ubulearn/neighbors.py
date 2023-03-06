import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import check_random_state, check_X_y, check_array
from sklearn.base import clone as skclone
from sklearn.exceptions import FitFailedWarning
import warnings


class DisturbingNeighborsClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator, knn=KNeighborsClassifier(n_neighbors=1), disturbing_instances=10, random_state=None):
        """
        Maudes, J., Rodríguez, J. J., & García-Osorio, C. (2009). 
        Disturbing neighbors diversity for decision forests. 
        In Applications of supervised and unsupervised ensemble methods (pp. 113-133).
        Springer, Berlin, Heidelberg.

        Parameters
        ----------
        base_estimator : ClassifierMixin
            Base estimator to be used for fitting the data.
        knn : kneighbors classifier, optional
            KNN Classifier or similar with `kneighbors` method, by default KNeighborsClassifier(n_neighbors=1)
        disturbing_instances : int, optional
            numer of instanses to choose as disturbed neighbors , by default 10
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None

        Raises
        ------
        AttributeError
            `knn` estimator must have a kneighbors method
        """
        self.base_estimator = base_estimator
        self.knn = knn
        self.disturbing_instances = disturbing_instances
        if "kneighbors" not in dir(self.knn):
            raise AttributeError(
                "`knn` estimator must have a kneighbors method")
        self.random_state = random_state
        self._estimator_type = "classifier"

    def __increase_features(self, X):

        nearest_class = self.nn1_.predict(X)
        nearest_neighbor = self.one_hot_.transform(
            self.nn1_.kneighbors(X, return_distance=False)).astype(bool)

        return np.concatenate((X, nearest_class.reshape(-1, 1), nearest_neighbor), axis=1)

    def fit(self, X, y, sample_weight=None, disturbing_instances=None, **fit_params):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels)
        sample_weight : array_like of shape (n_samples,) , optional
            Sample weights. If None, then samples are equally weighted. The behavior depends on the base estimator chose, by default None
        disturbing_instances : int {list, tuple, array}, optional
            list of instances to be used as disturbing neighbors, if None they are chosen randomly, by default None
            
        Returns
        -------
        self : DisturbingNeighborsClassifier
            Fitted estimator.
        """

        self.base_estimator_ = skclone(self.base_estimator)
        self.nn1_ = skclone(self.knn)
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        rs = check_random_state(self.random_state)
        self.encoder_ = LabelEncoder()
        y = self.encoder_.fit_transform(y)

        # Select c random features
        instances = X.shape[0]
        c = self.disturbing_instances if disturbing_instances is None else len(disturbing_instances)
        
        if disturbing_instances is None:
            try:
                self.instances_ = rs.choice(instances, c, replace=False)
            except Exception:
                self.instances_ = rs.choice(instances, c, replace=True)
                warnings.warn(
                    f"Not enough instances to select {c} random features. Replacing some instances.", FitFailedWarning)
        else:
            # Check if array is valid
            disturbing_instances = check_array(disturbing_instances, ensure_2d=False, dtype="int")
            if max(disturbing_instances) > instances:
                raise ValueError(f"disturbing_instances ({disturbing_instances}) has values greater than the number of instances ({instances})")
            self.instances_ = disturbing_instances

        self.one_hot_ = OneHotEncoder(sparse=False)
        self.one_hot_.fit(np.arange(c).reshape(-1, 1))

        # Create 1NN classifier
        self.nn1_ = self.nn1_.fit(X[self.instances_, :], y[self.instances_])
        X_large = self.__increase_features(X)

        self.base_estimator_.fit(X_large, y, sample_weight=sample_weight, **fit_params)
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
        """
        X_large = self.__increase_features(X)

        return self.base_estimator_.predict_proba(X_large)

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted classes.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
