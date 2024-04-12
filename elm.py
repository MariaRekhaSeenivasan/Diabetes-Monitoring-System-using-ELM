
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ELM():
    def __init__(self, hidden_units, activation_fn, x, y, one_hot=True, population_size=50):
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.x = x
        self.y = y
        self.population_size = population_size
        self.class_num = np.unique(self.y).shape[0] 
        self.elm_type = 'clf'
        self.one_hot = one_hot
        if self.elm_type == 'clf' and self.one_hot:
            self.one_hot_label = np.zeros((self.y.shape[0], self.class_num))
            for i in range(self.y.shape[0]):
                self.one_hot_label[i, int(self.y[i])] = 1
        # self.W = np.random.normal(loc=-1, scale=1, size=(self.hidden_units, self.x.shape[1]))
        # self.b = np.random.normal(loc=-1, scale=1, size=(self.hidden_units, 1))
        np.random.seed(42)
        self.W=np.random.randn(hidden_units,x.shape[1])
        self.b=np.random.randn(hidden_units)
        self.b = self.b[:, np.newaxis]  # or self.b = self.b.reshape(-1, 1)

    def input2hidden(self, x):
        self.temH = np.dot(self.W, x.T) + self.b
        self.H = 1 / (1 + np.exp(-self.temH))
        return self.temH
    
    def hidden2output(self, H):
        self.output = np.dot(self.H.T, self.beta)
        return self.output
    
    def fit(self):
        self.H = self.input2hidden(self.x)
        if self.elm_type == 'clf':
            if self.one_hot:
                self.y_temp = self.one_hot_label
            else:
                self.y_temp = self.y
        self.beta = np.dot(np.linalg.pinv(self.H.T), self.y_temp)
        self.result = self.hidden2output(self.H)
        self.result = np.exp(self.result) / np.sum(np.exp(self.result), axis=1).reshape(-1, 1)
        self.y_ = np.argmax(self.result, axis=1)
        self.correct = np.sum(self.y_ == self.y)
        self.train_score = self.correct / self.y.shape[0]

        return self.beta, self.train_score
    
    def predict(self, x):
        self.H = self.input2hidden(x)
        self.y_ = self.hidden2output(self.H.T)
        self.y_ = np.argmax(self.y_, axis=1)
        return self.y_
    
    def score(self, x, y):
        self.prediction = self.predict(x)
        if self.elm_type == 'clf':
            self.correct = np.sum(self.prediction == y)
            self.test_score = self.correct / y.shape[0]
            
        return self.test_score