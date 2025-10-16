import numpy as np

class MLP:
    def __init__(self, learningRate, mode):
        assert mode == "Classifier" or mode == "Regression", "unknown mode"
        self.mode = mode
        self.layers = []
        self.learningRate = learningRate

    def fit(self, x, y, epochs, noOfHiddenLayers, layerSizes):
        input_dim = x.shape[1]
        self.layers = []

        for i in range(noOfHiddenLayers):
            layer_input_dim = input_dim if i == 0 else layerSizes[i - 1]
            self.layers.append(Layer(layer_input_dim, layerSizes[i]))

        self.layers.append(Layer(layerSizes[-1], y.shape[1]))

        for i in range(epochs):
            output, activations, pre_activations = self.__forward(x)
            self.__backward(output, y, activations, pre_activations)

    def __forward(self, x):
        activations = [x]
        pre_activations = []

        for layer in self.layers[:-1]:
            z = activations[-1] @ layer.weights + layer.bias
            a = self.__relu(z)
            pre_activations.append(z)
            activations.append(a)

        if self.mode == "Classifier":
          z = activations[-1] @ self.layers[-1].weights + self.layers[-1].bias
          a = self.__softmax(z)
          pre_activations.append(z)
          activations.append(a)
          return a, activations, pre_activations
        
        else:
            z = activations[-1] @ self.layers[-1].weights + self.layers[-1].bias
            pre_activations.append(z)
            return z, activations, pre_activations

        

    def __backward(self, output, y, activations, pre_activations):
        error = output - y

        for i in reversed(range(len(self.layers))):
            a_prev = activations[i]
            z = pre_activations[i]

            if i != len(self.layers) - 1:
                error = error @ self.layers[i + 1].weights.T
                error *= self.__reluDerivative(z)

            gradientWeights = (a_prev.T @ error) / a_prev.shape[0]
            gradientBias = np.mean(error, axis=0, keepdims=True)

            self.layers[i].weights -= self.learningRate * gradientWeights
            self.layers[i].bias -= self.learningRate * gradientBias

    def __relu(self, x):
        return np.maximum(0, x)

    def __reluDerivative(self, x):
        return (x > 0).astype(float)

    def __softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def predict(self, x):
        output, _, _ = self.__forward(x)
        if self.mode == "Classifier":
          return np.argmax(output, axis=1) 
        else:
            return output

class Layer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))
