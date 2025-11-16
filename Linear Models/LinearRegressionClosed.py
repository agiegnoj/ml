import numpy as np

class LinearRegressionClosed:
    def __init__(self):
        self.beta = None
    
    def fit (self, X, y):
        
        assert isinstance(X, np.ndarray) and (X.ndim == 2), "wrong input, needs to be numpy array"
        assert isinstance(y, np.ndarray) and (y.ndim == 1), "wrong input, needs to be numpy array"

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        X_t = X.T
        product = X_t @ X

        if (np.linalg.matrix_rank(product) == product.shape[0]):
            inverseProduct = np.linalg.inv(product)
            
        else:
            inverseProduct = np.linalg.pinv(product)

        self.beta = inverseProduct @ X_t @ y
        
    
    def predict(self, x):
        assert self.beta is not None and isinstance(x, np.ndarray), "needs fitting"
        if x.ndim == 1:
           x = x.reshape(1, -1)
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x @ self.beta 
