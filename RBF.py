from collections import defaultdict
from sklearn.externals.funcsigs import signature
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import numpy as np


class RBFN(object):
    def __init__(self, basis='rbf', p_norm=2, sigma=0.5):
        self.basis = basis
        self.p = p_norm
        self.sigma = sigma
    
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])
    
    # Return the parameters of RBFN
    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    # Set the parameters of RBFN with corresponding dictionary
    def set_params(self, **params):
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
    
    # Get the centers by clustering the data points
    def _get_centers(self, x):
        brc = Birch()
        brc.fit(x)
        brc.predict(x)
        return brc.subcluster_centers_
    
    # The basis function (other basis functions will be added soon)
    def _basis_function(self, xi, center):
        """
        :param x: a vector (1*m), denotes one data point
        :param center: a vector (1*m), denotes one center point
        :return:
        """
        norm = np.linalg.norm(xi - center, ord=self.p)
        if self.basis == 'rbf':
            res = np.exp(-np.power(norm / self.sigma, 2) * .5)

        return res
    
    # Calculate the matrix according to basis function
    def _calculate_G(self, x):
        G = np.zeros((x.shape[0], self.centers.shape[0]))
        for i in range(x.shape[0]):
            for j in range(self.centers.shape[0]):
                G[i, j] = self._basis_function(x[i], self.centers[j])
        return G

    # Fit a rbfn model (calculate the corresponding weight of centers)
    def fit(self, x, y):
        x = np.asarray(x)
        if len(x.shape) == 1:
            self.centers = self._get_centers([x]).reshape(-1, 1)
        else:
            self.centers = self._get_centers(x)
        y = np.asarray(y)

        G = self._calculate_G(x)
        print(G.shape, self.centers.shape)
        self.weight = np.dot(np.linalg.pinv(G), y)
    
    # Predict the output according to the trained model.
    def predict(self, x):
        x = np.asarray(x)
        G = self._calculate_G(x)
        return np.dot(G, self.weight)


NUM_SAMPLES = 100
X = np.random.uniform(0., 1., NUM_SAMPLES)
X = np.sort(X, axis=0)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
y = np.cos(2 * np.pi * X) + noise

rbf = RBFN().set_params(**{'p_norm': 1, 'sigma': 1})
rbf.fit(X, y)
p = rbf.predict(X)
print(p)
plt.plot(X, y, '-o', label='true')
plt.plot(X, p, '-o', label='RBFN')
plt.legend()

plt.tight_layout()
plt.show()
