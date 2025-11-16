import numpy as np

class LogisticRegression:
    def __init__ (self, learningRate, tolerance= 1e-6):
        self.weights = None
        self.bias = None
        self.learningRate = learningRate
        self.tolerance = tolerance
    
    def fit (self, X, y, epochs):
        assert isinstance(X, np.ndarray) and (X.ndim == 2), "wrong input, needs to be numpy array"
        assert isinstance(y, np.ndarray) and (y.ndim == 1), "wrong input, needs to be numpy array"

        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        functionValue = X @ self.weights + self.bias
        sigmoid = self.__sigmoid(functionValue)
        currentLoss = 1e6
              
        for i in range (epochs):
            
            newWeights = self.weights - self.learningRate * self.__weightsGradient(X, y, sigmoid)
            newBias = self.bias - self.learningRate * self.__biasGradient(X, y, sigmoid)

            functionValue = X @ newWeights + newBias
            sigmoid = self.__sigmoid(functionValue)
            newLoss = self.__loss(sigmoid, y) 

            if abs(currentLoss - newLoss) < self.tolerance:
                break

            currentLoss = newLoss
            
    def predict(self, X):
        pred = 1 / (1 + np.exp(-(X @ self.weights + self.bias)))
        return (pred > 0.5).astype(int)

    def __sigmoid(self, functionValue):

        return 1 / (1+ np.exp(-(functionValue)))

               
    def __weightsGradient(self, X, y, sigmoid):
        m = X.shape[0]
        return (X.T @ (sigmoid-y))/m
    
    def __biasGradient(self, X, y, sigmoid):
        m = X.shape[0]
        return np.sum((sigmoid-y))/m
        
    def __loss(self, yPred, yTrue):
        yZeroLoss = yTrue * np.log(yPred + 1e-9)
        yOneLoss = (1-yTrue) * np.log(1 - yPred + 1e-9)
        
        return -np.mean(yZeroLoss + yOneLoss)
    