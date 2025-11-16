import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier
from multiprocessing import Process, Manager, freeze_support

class RandomForest:
    def __init__ (self, mode= "Classifier"):
        assert mode == "Classifier" or mode == "Regression", "unknown mode"
        self.trees = Manager().list()
        self.mode = mode

    def fit(self, noOfTrees, X, y, maxDepthPerTree, minElementsPerNode, parallelComputations=4):
        self.masks = []
        batchSize = noOfTrees//parallelComputations
        processes = []
        
        for i in range (parallelComputations):
            p = Process(target=buildTrees, args=(X, y, self.trees, batchSize, maxDepthPerTree, minElementsPerNode))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
    def predict(self, x):
        predictions = np.zeros(len(self.trees))

        for i in range (len(self.trees)):
            predictions[i] = self.trees[i].predict(x)

        predictions = np.array(predictions, dtype=int)
        if self.mode == "Classifier":
           return np.bincount(predictions).argmax()
        elif self.mode == "Regression":
            return np.mean(predictions)


def buildTrees(X, y, trees, maxDepthPerTree, noOfTrees, minElementsPerNode):
    for i in range (noOfTrees):
        samples= X.shape[0]
        indices = np.random.choice(samples, size=samples, replace=True)
        attributes = X[indices]
        values = y[indices]

        dTree = DecisionTreeClassifier(maxDepthPerTree, minElementsPerNode)
        dTree.fit(attributes, values, True)
        trees.append(dTree)

if __name__ == "__main__":
    freeze_support()
