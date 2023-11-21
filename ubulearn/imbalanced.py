# Author: José Miguel Ramírez-Sanz
# @: jmrsanz@ubu.es
# github: https://github.com/Josemi
# Algorithm: Juanjo José Rodríguez, José-Francico Díez-Pastor, Álvar Arnaiz-González, Ludmila Kuncheva
# SOURCE: https://github.com/Josemi/MultiRandBal_Python
# Adapted to scikit-learn by José Luis Garrido Labrador


from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
import random as rnd
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd

class MultiRandBalClassifier(BaseEstimator, ClassifierMixin):
    """
    MultiRandBal is a class that implements the Multi-Random Balanced Sampling (MultiRandBal) algorithm.

    José F. Díez-Pastor, Juan J. Rodríguez, César García-Osorio, Ludmila I. Kuncheva,
    Random Balance: Ensembles of variable priors classifiers for imbalanced data,
    Knowledge-Based Systems, Volume 85, 2015, Pages 96-111,
    ISSN 0950-7051,
    https://doi.org/10.1016/j.knosys.2015.04.022.

    Juan J. Rodríguez, José-Francisco Díez-Pastor, Álvar Arnaiz-González, Ludmila I. Kuncheva,
    Random Balance ensembles for multiclass imbalance learning,
    Knowledge-Based Systems, Volume 193, 2020, 105434, 
    ISSN 0950-7051,
    https://doi.org/10.1016/j.knosys.2019.105434.

    Parameters
    ----------
    n_estimators : int
        The number of estimators to use in the ensemble.
    base_estimator : object
        The base estimator to fit on random subsets of the dataset.
    oversampler : object
        The oversampler to use.
    undersampler : object
        The undersampler to use.
    min_samples : int
        The minimum number of samples on any class.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;

    Attributes
    ----------
    estimators_ : list of objects
        The collection of fitted sub-estimators.
    classes_ : array, shape = [n_classes]
        The classes labels.
    n_classes_ : int
        The number of classes.
    n_samples_ : list of int
        The number of samples in each class.
    n_samples_ttl_ : int
        The total number of samples.
    """


    def __init__(self, n_estimators=100, base_estimator=DecisionTreeClassifier, oversampler = SMOTE, undersampler = RandomUnderSampler,min_samples=2, random_state=None):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.oversampler = oversampler
        self.undersampler = undersampler
        self.min_samples = min_samples
        self.random_state = random_state
        #Create estimators
        self.estimators_ = []

        for i in range(self.n_estimators):
            self.estimators_.append(self.base_estimator(random_state=self.random_state))

        
    def _get_random_balance(self,X,y):
        """
        Get a random balanced dataset.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        
        Returns
        -------
        X_bal : array-like, shape = [n_samples_bal, n_features]
            The balanced training input samples.
        y_bal : array-like, shape = [n_samples_bal]
            The balanced target values.
        """
        #final data

        data = pd.DataFrame(X)
        y_df = pd.DataFrame(y)
        data[y_df.columns[0]] = y
        proportion_list = []
        for c in range(self.n_classes_):
            proportion_list.append(rnd.uniform(0,1))

        w = sum(proportion_list)

        undersample_dict = {}
        oversample_dict = {}
        for c in range(self.n_classes_):
            proportion = max(round(self.n_samples_ttl_*proportion_list[c]/w), self.min_samples)

            if proportion > self.n_samples_[c]:
                #oversample
                oversample_dict[c] = proportion
            elif proportion < self.n_samples_[c]:
                #undersample
                undersample_dict[c] = proportion

        smote = self.oversampler(random_state=self.random_state, sampling_strategy=oversample_dict)

        X_bal,y_bal= smote.fit_resample(X, y)    

        rus = self.undersampler(sampling_strategy=undersample_dict, random_state=self.random_state)
        X_bal,y_bal = rus.fit_resample(X_bal,y_bal)

        return X_bal,y_bal
        

    def fit(self, X, y):
        """
        Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        #Get classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_samples_ = [len(y[y == c]) for c in self.classes_]
        self.n_samples_ttl_ = len(y)

        #Fit estimators
        for estimator in self.estimators_:
             #Get balanced dataset
            X_bal,y_bal = self._get_random_balance(X,y)
            estimator.fit(X_bal,y_bal)

        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.
        
        Returns
        -------
        y_pred : array, shape = [n_samples]
            The predicted classes.
        """
        #Predict
        y_pred = []
        for estimator in self.estimators_:
            y_pred.append(estimator.predict(X))

        #Get majority vote
        y_pred = np.array(y_pred).T
        y_pred = [Counter(y_pred[i]).most_common(1)[0][0] for i in range(len(y_pred))]

        return np.array(y_pred)
        
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.
        
        Returns
        -------
        y_pred_proba : array, shape = [n_samples, n_classes]
            The predicted class probabilities.
        """
        #Predict
        y_pred_proba = []
        for estimator in self.estimators_:
            y_pred_proba.append(estimator.predict_proba(X))

        
        #Get average
        y_pred_proba = np.array(y_pred_proba)
        y_pred_proba = np.mean(y_pred_proba, axis=0)

        return np.array(y_pred_proba)

            