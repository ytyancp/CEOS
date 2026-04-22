import warnings
from ceos import CEOS
from ceos_fm import FM
from ceos_cs import CS
from ceos_os import OS
from ceos_us import US
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")


class Error(Exception):
    pass


class ResampleClassifier(object):
    """
    Re-sampling methods for imbalance classification, based on imblearn python package.
    imblearn url: https://github.com/scikit-learn-contrib/imbalanced-learn
    Hyper-parameters:
        base_estimator : scikit-learn classifier object
            optional (default=DecisionTreeClassifier)
            The base estimator used for training after re-sampling
    """

    def __init__(self, base_estimator=DecisionTreeClassifier()):
        self.base_estimator = base_estimator

    def predict(self, X):
        return self.base_estimator.predict(X)

    def fit(self, X, y, method):
        """
        method: String
            The method used to perform re-sampling
        """
        if method == "CEOS":
            sampler = CEOS()
        elif method == "CEOS_CS":
            sampler = CS()
        elif method == "CEOS_US":
            sampler = US()
        elif method == "CEOS_OS":
            sampler = OS()
        elif method == "CEOS_FM":
            sampler = FM()
        elif method == "Original":
            sampler = None
        else:
            raise Error('No such method support: {}'.format(method))

        if method != 'Original':
            X_train, y_train = sampler.fit_resample(X, y)
        else:
            X_train, y_train = X, y

        self.base_estimator.fit(X_train, y_train)
