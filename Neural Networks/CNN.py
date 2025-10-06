import numpy as np

class CNN:
    def __init__(self, noOfFilters, filterSize, stepSize, pFilterSize, pStepSize, learningRate, mode):
        assert mode == "Classifier" or mode == "Regression", "unknown mode"
        self.mode = mode
        self.noOfFilters = noOfFilters
        self.filterSize = filterSize
        self.stepSize = stepSize
        self.learningRate = learningRate
        self.pFilterSize = pFilterSize
        self.pStepSize = pStepSize
        self.weights = None
        self.bias = None

    def fit(self, images, classes, epochs):
        self.fl = FilterLayer(self.noOfFilters, self.filterSize, self.stepSize)
        self.pl = PoolingLayer(self.pFilterSize, self.pStepSize)
        
        self.uniqueClasses = len(np.unique(classes))

        for epoch in range(epochs):
            for idx in range(len(images)):
                pred, error, flattened, preFlattenedDimensions = self.__forward(images[idx], classes[idx], len(set(classes)))
                self.__backward(pred, error, flattened, preFlattenedDimensions)

    def predict(self, image):
        pred, _, _, _ = self.__forward(image, 1, self.uniqueClasses)
        return np.argmax(pred, axis=0)

    def __forward(self, image, actualClass, numClasses):
        f = self.fl.forward(image)
        f = self.pl.forward(f)

        preFlattenDimensions = f.shape
        actualClassVec = np.eye(numClasses)[actualClass]
        flattened = f.flatten()

        if self.weights is None:
            self.weights = np.random.rand(numClasses, len(flattened))
            self.bias = np.random.rand(numClasses)

        prediction = self.__softMax(flattened) if self.mode == "Classifier" else np.dot(self.weights, flattened) + self.bias
        error = prediction - actualClassVec

        return prediction, error, flattened, preFlattenDimensions

    def __backward(self, pred, error, flattened, dimensionsPreFlattened):
        dW = np.outer(error, flattened)
        dB = error
        self.weights -= self.learningRate * dW
        self.bias -= self.learningRate * error

        deltaFlattened = np.dot(self.weights.T, error)
        deltaPooled = np.reshape(deltaFlattened, dimensionsPreFlattened)

        deltaRelu = self.pl.backward(deltaPooled)
        self.fl.backward(deltaRelu)

    def __softMax(self, x):
        totals = np.dot(self.weights, x) + self.bias
        totals -= np.max(totals)
        expX = np.exp(totals)
        return expX / np.sum(expX)


class FilterLayer:
    def __init__(self, noOfFilters, filterSize, stepSize):
        self.filters = [np.random.rand(filterSize, filterSize)*0.1 for _ in range(noOfFilters)]
        self.filterSize = filterSize
        self.stepSize = stepSize

    def forward(self, input):
        self.prefiltered = input
        self.activated = [self.__relu(self.__applyFilter(input, f)) for f in self.filters]
        return np.array(self.activated)

    def __applyFilter(self, input, filter):
        size = (input.shape[0] - self.filterSize) // self.stepSize + 1
        output = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                region = input[i*self.stepSize:i*self.stepSize+self.filterSize,
                               j*self.stepSize:j*self.stepSize+self.filterSize]
                output[i, j] = np.sum(region * filter)

        return output

    def __relu(self, input):
        return np.maximum(input, 0)

    def backward(self, deltaRelu):
        for idx, (filter, activation) in enumerate(zip(self.filters, self.activated)):
            reluMask = activation > 0
            delta = deltaRelu[idx] * reluMask

            size = (self.prefiltered.shape[0] - self.filterSize) // self.stepSize + 1
            for i in range(size):
                for j in range(size):
                    region = self.prefiltered[i*self.stepSize:i*self.stepSize+self.filterSize,
                                              j*self.stepSize:j*self.stepSize+self.filterSize]
                    filter += delta[i, j] * region


class PoolingLayer:
    def __init__(self, filterSize, stepSize):
        self.filterSize = filterSize
        self.stepSize = stepSize

    def forward(self, input):
        self.memo = []
        self.reluInput = input
        pooled = []

        for m in input:
            pooledMap, memoMap = self.__applyPooling(m)
            pooled.append(pooledMap)
            self.memo.append(memoMap)

        return np.array(pooled)

    def __applyPooling(self, input):
        if input.shape[0] < self.filterSize or input.shape[1] < self.filterSize:
            return input, np.zeros_like(input, dtype=object)

        size = (input.shape[0] - self.filterSize) // self.stepSize + 1
        output = np.zeros((size, size))
        memo = np.zeros((size, size), dtype=object)

        for i in range(size):
            for j in range(size):
                region = input[i*self.stepSize:i*self.stepSize+self.filterSize,
                               j*self.stepSize:j*self.stepSize+self.filterSize]
                maxIndex = np.argmax(region)
                maxPos = np.unravel_index(maxIndex, region.shape)
                memo[i, j] = (i*self.stepSize + maxPos[0], j*self.stepSize + maxPos[1])
                output[i, j] = region[maxPos]

        return output, memo

    def backward(self, deltaPooled):
        deltaRelu = np.zeros_like(self.reluInput)

        for idx, memoMap in enumerate(self.memo):
            for i in range(memoMap.shape[0]):
                for j in range(memoMap.shape[1]):
                    maxPos = memoMap[i, j]
                    deltaRelu[idx][maxPos] = deltaPooled[idx][i, j]

        return deltaRelu
                
          