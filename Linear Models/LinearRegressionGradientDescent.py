import numpy as np

class LinearRegressionGradientDescent:
    
    def __init__(self, learningRate, tolerance= 1e-6):
        self.beta = None
        self.bias = None
        self.learningRate = learningRate
        self.tolerance = tolerance

    def fit(self, X, y, epochs):
        assert isinstance(X, np.ndarray) and (X.ndim == 2), "wrong input, needs to be numpy array"
        assert isinstance(y, np.ndarray) and (y.ndim == 1), "wrong input, needs to be numpy array"

        weights = np.random.rand(X.shape[1])
        bias = np.random.rand()
        yPred = self.__yPred(X, weights, bias)
        currentLoss = self.__loss(y, yPred)
        
        for i in range (epochs):
            
            newWeights = weights-(self.learningRate*self.__weightsGradient(yPred, y, X ))
            newBias = bias - (self.learningRate*self.__biasGradient(yPred, y))
            newLoss = self.__loss(y, self.__yPred(X, newWeights, newBias))
            weights = newWeights
            bias = newBias
            self.beta = weights
            self.bias  = bias
            if abs (currentLoss - newLoss) < self.tolerance:
                break
            currentLoss = newLoss
            yPred = self.__yPred(X, self.beta, self.bias)
                     
            
    def predict(self, x):
        print (self.beta)
        assert self.beta is not None, "needs fitting"
        return x@self.beta+self.bias
        
    def __loss(self, y, yPred):
        return np.mean((yPred-y)**2)
    
    def __yPred(self, X, weights, bias):          
        return X @ weights + bias
    
    def __weightsGradient(self, yPred, y, X):
        return 2*(X.T @ (yPred-y))/len(X)
             
    def __biasGradient(self, yPred, y):
        return 2*np.sum((yPred-y))/len(y)
    