import numpy as np

class SVMLinear:
    def __init__(self, learningRate, C = 1):
        self.weights = None
        self.bias = None
        self.learningRate = learningRate
        self.C = C
    
    def fit(self, X, y, epochs):
        self.weights = np.ones(X.shape[1])
        self.bias = 0

        for i in range (epochs):
            gradientWeights, gradientBias = self.__gradients(X, y)
            self.weights -= self.learningRate*gradientWeights
            self.bias -= self.learningRate* gradientBias

        
    def predict(self, x):
        assert isinstance(x, np.ndarray), "wrong input, needs to be numpy array"
        assert self.weights is not None, "needs fitting"
        return np.sign(self.weights @ x + self.bias)

    def __gradients(self, X, y):
        margins = y *(X @ self.weights + self.bias)
        mask = margins < 1
   
        gradientWeights = self.weights - self.C * np.sum((y[mask, np.newaxis] * X[mask]), axis=0)
        gradientBias = -self.C*np.sum(y[mask])
   
        return gradientWeights, gradientBias
