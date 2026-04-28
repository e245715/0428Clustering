import numpy as np

class LinearRegression:
    X = None
    theta = None
    Y = None
    
    def fit(self,X,Y):
        temp = np.linalg.inv(np.dot(X.T,X))
        self.theta = np.dot(np.dot(temp,X.T),Y)
    
    def predict(self,X):
        return np.dot(X,self.theta)
    
    def score(self,X,Y):
        error = self.predict(X) - Y
        return (error**2).sum()