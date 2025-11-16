import numpy as np
from collections import Counter

class DecisionTreeRegression:

    def __init__(self, maxDepth, minElementsPerNode, minErrorDecrease = 1e-3):
        self.maxDepth = maxDepth
        self.minElementsPerNode = minElementsPerNode
        self.root = None
        self.minErrorDecrease = minErrorDecrease
        self.mse_cache = {}

    def fit(self, Attributes, values):
        self.root = Node()
        nodeValues = np.array(values)
        sortedIndices = self.__sortedIndices(Attributes)
        unusedRows = [i for i in range(len(Attributes))]
        self.__buildTree(Attributes, unusedRows, sortedIndices, values, nodeValues, self.root, 0)

    def predict(self, attributes):
        currentNode = self.root
        while currentNode.getPrediction() is None:
            splitValue, columnIndex = currentNode.getCondition()
            if attributes[columnIndex] < splitValue:
                currentNode = currentNode.getLeftNode()
            else:
                currentNode = currentNode.getRightNode()
        return currentNode.getPrediction()

    def __buildTree(self, Attributes, unusedRows, sortedIndices, values, nodeValues, currentNode, depth):
        if (depth >= self.maxDepth or len(unusedRows) < self.minElementsPerNode):
            prediction = np.mean(nodeValues)
            currentNode.setPrediction(prediction)
            return
        
        currentError = self.__mse(nodeValues)
        minError = float('inf')
        columnIndex = None
        splitValue = None

        for i in range(Attributes.shape[1]):
            error, split = self.__attributeSplitPoint(Attributes, sortedIndices, unusedRows, values, i)
            if error < minError:
                columnIndex = i
                splitValue = split

        if  (currentError-minError)< self.minErrorDecrease or splitValue is None or columnIndex is None:
            prediction = np.mean(nodeValues)
            currentNode.setPrediction(prediction)
            return

        currentNode.setCondition(splitValue, columnIndex)

        unusedRowsLeft = []
        unusedRowsRight = []
        newValues1 = []
        newValues2 = []

        for i in unusedRows:
            if Attributes[i][columnIndex] < splitValue:
                unusedRowsLeft.append(i)
                newValues1.append(values[i])
            else:
                unusedRowsRight.append(i)
                newValues2.append(values[i])

        if len(unusedRowsLeft) > 0:
            leftNode = Node()
            currentNode.setLeftNode(leftNode)
            self.__buildTree(Attributes, unusedRowsLeft, sortedIndices, values, np.array(newValues1), leftNode, depth + 1)

        if len(unusedRowsRight) > 0:
            rightNode = Node()
            currentNode.setRightNode(rightNode)
            self.__buildTree(Attributes, unusedRowsRight, sortedIndices, values, np.array(newValues2), rightNode, depth + 1)

    def __attributeSplitPoint(self, Attributes, sortedIndices, unusedRows, values, columnIndex):
        sorted_idx = sortedIndices[columnIndex]
        unusedSet = set(unusedRows)
        sorted_idx = [i for i in sorted_idx if i in unusedSet]

        if len(sorted_idx) < 2:
            return 1, None

        column = Attributes[sorted_idx, columnIndex]
        values_sorted = values[sorted_idx]

        best_error = float('inf')
        best_split = None

        for i in range(len(column) - 1):
            if column[i] == column[i + 1]:
                continue
            split_val = (column[i] + column[i + 1]) / 2
            left = values_sorted[:i + 1]
            right = values_sorted[i + 1:]
            total = len(values_sorted)
            error = (len(left)*self.__mse(left) + len(right)*self.__mse(right))/total
            if error < best_error:
                best_error = error
                best_split = split_val

        return best_error, best_split

    def __mse(self, y):
        key = y.tobytes()
        if (key not in self.mse_cache):
            self.mse_cache[key] = np.mean((y - np.mean(y))**2)
            
        return self.mse_cache[key]
        
        

    def __sortedIndices(self, att):
        sorted_indices = []
        for i in range(att.shape[1]):
            sorted_indices.append(np.argsort(att[:, i]))
        return sorted_indices
    
class Node:
    def __init__(self):
        self.left_node = None
        self.right_node = None
        self.condition = None  
        self.prediction = None

    def getLeftNode(self):
        return self.left_node

    def getRightNode(self):
        return self.right_node

    def getCondition(self):
        return self.condition

    def getPrediction(self):
        return self.prediction

    def setLeftNode(self, ln):
        self.left_node = ln

    def setRightNode(self, rn):
        self.right_node = rn

    def setCondition(self, sp, index):
        self.condition = (sp, index)

    def setPrediction(self, p):
        self.prediction = p
