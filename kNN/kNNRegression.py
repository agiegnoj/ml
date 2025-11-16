import numpy as np
from multiprocessing import Process, Manager, freeze_support

class kNNRegression:  
    def __init__ (self):
            self.X = None
            self.labels = None

    def fit (self, X, labels):
            self.X = X
            self.labels = labels

                             
    def predict (self, x, k, parallelCalculations = 2):          
           
           manager = Manager()
           distances = manager.list([0.0] * self.X.shape[0])
           procecess = []
           batchSize = self.X.shape[0] 

           for i in range (parallelCalculations):
                  
                  start = i * batchSize
                  end = (i + 1) * batchSize if i < parallelCalculations - 1 else self.X.shape[0]
                  p = Process(target=computeDistances, args=(self.X, x, start, end, distances))
                  procecess.append(p)

           for p in procecess:
                  p.start()
           for p in procecess:
                  p.join()
       
           indices = np.argsort(np.array(distances))

           sum = 0
      
           for i in range (k):
                  sum += self.labels[indices[i]]
                 
           return sum/k
        
        
def computeDistances(X, x1, startIndex, endIndex, distances):  
            for i in range (startIndex, endIndex):
                  x2 = X[i]
                  d = 1-x1.T @ x2/(np.sqrt(np.sum(x1**2))*np.sqrt(np.sum(x2**2)))
                  distances[i] = d       
            

if __name__ == "__main__":
    freeze_support()
