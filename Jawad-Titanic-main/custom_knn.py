import numpy as np
import pandas as pd

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X.values
        self.y_train = y.values

    def predict_one(self, x):
        distances = []
        for i in range(len(self.X_train)):
            dist = euclidean_distance(self.X_train[i], x)
            distances.append((dist, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_neighbors = distances[:self.k]

        labels = [label for _, label in k_neighbors]
        prediction = max(set(labels), key=labels.count)
        return prediction

    def predict(self, X):
        predictions = []
        X_np = X.values  # avoids KeyError
        for i in range(len(X_np)):
            predictions.append(self.predict_one(X_np[i]))
        return np.array(predictions)
