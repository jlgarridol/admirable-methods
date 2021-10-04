from os import replace
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.feature_selection import VarianceThreshold
import sklearn.utils
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone as skclone



class Rotation(TransformerMixin, BaseEstimator):
    
    def __init__(self, group_size=3, group_weight=.50, pca=PCA(svd_solver="full"), random_state=None):
        """Rotation Transformer.

        Join of subspaces of the dataset transfomed with PCA keeping the subspace dimensionality.

        The transformation is made with a subsamples of the subspace. The subsamples and the subspaces are made randomly.

        Args:
            group_size (int, optional): Number of the features for each subspace. If the number of features mod group size are not zero, then the last subspace are created with features used previously selected randomly.. Defaults to 3.
            group_weight (float, optional): Proportion of instances to be removed.. Defaults to .5.
            pca (PCA, optional): PCA configuration, the n_components will be overwritten. Defaults to PCA().
            random_state (None int or RandomState, optional): Random state for create subspaces and subsamples. Defaults to None.
        """
        self.group_size=group_size
        self.random_state=random_state
        self.group_weight=group_weight
        self.pca = pca        
        
    def fit(self, X, y=None):
        """Create a rotation.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (None): If not None it used for keep class proportion.
        """
        self.groups_ = []
        self.pcas_ = []

        rows, cols = X.shape
        cl = list(range(cols))
        random_state = sklearn.utils.check_random_state(self.random_state)

        random_state.shuffle(cl)  # Shuffle columns

        # Generate random subspaces bassed on before shuffle
        idx = 0
        while idx < len(cl):
            gr = []
            for i in range(self.group_size):
                if i+idx >= len(cl):
                    gr.append(random_state.choice(cl))
                else:
                    gr.append(cl[i+idx])
            
            self.groups_.append(gr)
            idx += self.group_size
        # End

        # Select a random subset for each group 
        groups_X = []
        for g in self.groups_:
            groups_X.append(X[:,g])
        groups_T = []
        for g in groups_X:
            # First remove a group weight.
            sub_g, sub_y = sklearn.utils.resample(g, y, replace=False, n_samples=int((1-self.group_weight)*rows), random_state=random_state.randint(100))
            sel = sklearn.utils.resample(sub_g, replace=True, n_samples=int(0.75*sub_g.shape[0]), random_state=random_state.randint(100), stratify=sub_y)
            # Se "barajean" las filas
            #random_state.shuffle(rl)
            #sel = int(self.group_weight*rows)
            groups_T.append(sel)
        
        # Una vez se tienen los objetos para entrenar entonces se crean los PCA
        for g in groups_T:
            p = skclone(self.pca)
            p.random_state = random_state.randint(100)
            p.n_componentes_ = self.group_size
            #p = PCA(self.group_size)
            p.fit(g)
            self.pcas_.append(p)
            
        # PCA        
        
            
    def transform(self, X):
        """Apply rotation to X.

        X rotated in each subspace and then the rotated subspaces are joined to create the global rotation of X.

        Args:
            X (array-like, shape (n_samples, n_features)): New data, where n_samples is the number of samples and n_features is the number of features.

        Returns:
            array-like, shape (n_samples, n_components): Transformed values.
        """
        if not "pcas_" in dir(self):
            raise NotFittedError("Fit before transform.")
        tformed = []
        for i in range(len(self.pcas_)):
            pca = self.pcas_[i]
            group = self.groups_[i]
            x_n = X[:,group]
            x_t = pca.transform(x_n)
            tformed.append(x_t)
            
        return np.concatenate(tformed,axis=1)
            
    def fit_transform(self, X, y=None):
        """Create the rotation of X and get the rotation of X.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (None): If not None it used for keep class proportion.

        Returns:
            array-like, shape (n_samples, n_components): Rotation of X.
        """
        self.fit(X, y)
        return self.transform(X)
            

class RotatedTree(ClassifierMixin, BaseEstimator):


    def __init__(self, base_estimator=DecisionTreeClassifier(), rotation=Rotation()):
        """Create a rotation and train a decision tree classifier.

        Args:
            base_estimator (object, optional): The base estimator to fit on rotation of the dataset. If None, then the base estimator is a decision tree. Defaults to DecisionTreeClassifier().
            rotation (Rotation, optional): The configured rotation transform. Defaults to Rotation().
        """
        self.base_estimator = skclone(base_estimator)
        self.rotation = skclone(rotation)

    def fit(self, X, y):
        """Fit the Rotated Tree model.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like, shape (n_samples,)): The target values.
        """
        X = self.rotation.fit_transform(X, y)
        self.base_estimator.fit(X,y)
        self.classes_=self.base_estimator.classes_

        return self

    def predict(self, X):
        """Predict class for X.

        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            ndarray of shape (n_samples,): The predicted classes.
        """
        X = self.rotation.transform(X)
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.


        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            array-like, shape (n_samples, n_features): The class probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.
        """
        X = self.rotation.transform(X)
        return self.base_estimator.predict_proba(X)

class RotationForestClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=10, min_group_size=3, max_group_size=3, rotation=Rotation(), random_state=None):
        """Create a ensemble of rotation trees for clasification.

        Args:
            base_estimator (object, optional): The base estimator to fit on each rotation of the dataset. If None, then the base estimator is a decision tree. Defaults to DecisionTreeClassifier().
            n_estimators (int, optional): Number of estimators in the ensemble. Defaults to 10.
            min_group_size (int, optional): Min group of the features for subspaces. Defaults to 3.
            max_group_size (int, optional): Max group of the features for subspaces. Defaults to 3.
            rotation (Rotation, optional): Configuration for a rotation. The random_state and group_size will be overwritten in each iteration. Defaults to Rotation().
            random_state (None int or RandomState, optional): Random state for create subspaces and subsamples. Defaults to None.
        """
        self.random_state=random_state
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.base_estimator = base_estimator
        self.rotation = rotation
        self.n_estimators = n_estimators

    def fit(self, X, y):
        """Fit the RotationForest model.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like, shape (n_samples,)): The target values.
        """
        random_state = sklearn.utils.check_random_state(self.random_state)

        self.estimators_ = []
        for _ in range(self.n_estimators):
            size = random_state.randint(self.min_group_size, self.max_group_size+1)
            rotation = skclone(self.rotation)
            rotation.group_size = size
            rotation.random_state = random_state.randint(100)
            tree = RotatedTree(self.base_estimator, rotation)
            self.estimators_.append(tree.fit(X, y))
        self.classes_ = self.estimators_[0].classes_
        return self
    
    def predict(self, X, **kwards):
        """Predict the classes of X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Array with predicted labels.
        """
        predicted_probabilitiy = self.predict_proba(X, **kwards)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X, **kwargs):
        """Predict class probabilities for X.

        The probability for each instance is the mean between all classifiers in ensemble.


        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            array-like, shape (n_samples, n_features): The class probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.
        """
        probas = []
        for t in self.estimators_:
            predicts = t.predict_proba(X, **kwargs)
            probas.append(predicts)
        probas = np.array(probas)
        return probas.mean(axis=0)
