import numpy as np

class CNN:
    # config = noOfFilters, filterSizes, stepSizes, pfilterSizes, pstepSizes (each as a list)
    
    def __init__(self, config, learningRate, mode, noOfFilterAndPoolingCombinations, filterLearningrate):
        assert mode in ["Classifier", "Regression"], "unknown mode"
        self.mode = mode
        self.noOfFilters, self.filterSizes, self.stepSizes, self.pfilterSizes, self.pstepSizes = config
        self.learningRate = learningRate
        self.weights = None
        self.bias = None
        self.layers = []
        for i in range(noOfFilterAndPoolingCombinations):
            self.layers.append(FilterLayer(self.noOfFilters[i], self.filterSizes[i], self.stepSizes[i], filterLearningrate))
            self.layers.append(PoolingLayer(self.pfilterSizes[i], self.pstepSizes[i]))
        

    def fit(self, images, classes, epochs):
        self.uniqueClasses = len(np.unique(classes))
        
        for epoch in range(epochs):
            print (epoch)
            
            for idx in range(len(images)):
                pred, error, flattened, preFlattenedDimensions = self.__forward(images[idx], classes[idx], self.uniqueClasses)
                self.__backward(error, flattened, preFlattenedDimensions)
                    

    def predict(self, image):
        pred, _, _, _ = self.__forward(image, 0, self.uniqueClasses)
        return np.argmax(pred, axis=0)

    def __forward(self, image, actualClass, numClasses):
        f = image
        for l in self.layers:
            f = l.forward(f)
        preFlattenDimensions = f.shape
        actualClassVec = np.eye(numClasses)[actualClass]
        flattened = f.flatten()
        if self.weights is None:
            self.weights = np.random.rand(numClasses, len(flattened)) * 0.1
            self.bias = np.random.rand(numClasses) * 0.1
        prediction = self.__softMax(flattened) if self.mode == "Classifier" else np.dot(self.weights, flattened) + self.bias
        error = prediction - actualClassVec
        return prediction, error, flattened, preFlattenDimensions

    def __backward(self, error, flattened, dimensionsPreFlattened):
        dW = np.outer(error, flattened)
        
        self.weights -= self.learningRate * dW
        self.bias -= self.learningRate * error
        deltaFlattened = np.dot(self.weights.T, error)
        deltaPooled = np.reshape(deltaFlattened, dimensionsPreFlattened)
        f = deltaPooled
        for l in reversed(self.layers):
            f = l.backward(f)

    def __softMax(self, x):
        totals = np.dot(self.weights, x) + self.bias
        totals -= np.max(totals)
        expX = np.exp(totals)
        return expX / np.sum(expX)


class FilterLayer:
    def __init__(self, noOfFilters, filterSize, stepSize, learningRate):
        self.filters = None
        self.noOfFilters = noOfFilters
        self.filterSize = filterSize
        self.stepSize = stepSize
        self.learningRate = learningRate

    def forward(self, input):
        if input.ndim == 2:
            input = np.expand_dims(input, axis=0)
        depth = input.shape[0]
        if self.filters is None:
            self.filters = [np.random.rand(depth, self.filterSize, self.filterSize)*0.1 for _ in range(self.noOfFilters)]
        self.prefiltered = input
        size = (input.shape[1] - self.filterSize) // self.stepSize + 1
        self.activated = np.array([self.__applyFilter(input, f, size)for f in self.filters])
        return np.maximum(0, self.activated)

    def backward(self, deltaRelu):
        grad_input = np.zeros_like(self.prefiltered)
        grad_filters = [np.zeros_like(f) for f in self.filters]
        size = (self.prefiltered.shape[1] - self.filterSize) // self.stepSize + 1
        
        for idx, (filter, activation) in enumerate(zip(self.filters, self.activated)):
            relu_mask = activation > 0
            delta = deltaRelu[idx] * relu_mask
            
            for i in range(size):
                for j in range(size):
                    region = self.prefiltered[:, i*self.stepSize:i*self.stepSize+self.filterSize,
                                              j*self.stepSize:j*self.stepSize+self.filterSize]
                    grad_filters[idx] += delta[i, j] * region
                    grad_input[:, i*self.stepSize:i*self.stepSize+self.filterSize,
                                  j*self.stepSize:j*self.stepSize+self.filterSize] += delta[i, j] * filter
            self.filters[idx] -= self.learningRate* grad_filters[idx]
        return grad_input


    def __applyFilter(self, input, filter, size):
        
        output = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                region = input[:, i*self.stepSize:i*self.stepSize+self.filterSize,
                                  j*self.stepSize:j*self.stepSize+self.filterSize]
                output[i, j] = np.sum(region * filter)

        return output


class PoolingLayer:
    def __init__(self, filterSizes, stepSizes):
        self.filterSizes = filterSizes
        self.stepSizes = stepSizes

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
        size = (input.shape[0] - self.filterSizes) // self.stepSizes + 1
        output = np.zeros((size, size))
        memo = np.zeros((size, size), dtype=object)
        for i in range(size):
            for j in range(size):
                region = input[i*self.stepSizes:i*self.stepSizes+self.filterSizes,
                               j*self.stepSizes:j*self.stepSizes+self.filterSizes]
                maxIndex = np.argmax(region)
                maxPos = np.unravel_index(maxIndex, region.shape)
                memo[i, j] = (i*self.stepSizes + maxPos[0], j*self.stepSizes + maxPos[1])
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
    
