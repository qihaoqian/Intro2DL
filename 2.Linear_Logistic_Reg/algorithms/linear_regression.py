"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        C = weights.shape[0]  
        self.w = weights  

        lambda_reg = self.weight_decay

        y_train_one_hot = np.zeros((N, C))
        y_train_one_hot[np.arange(N), y_train] = 1

        for epoch in range(self.epochs):

            scores = X_train @ self.w.T

            loss = np.sum((scores - y_train_one_hot) ** 2) / N + lambda_reg * np.sum(self.w ** 2)

            grad = (2 / N) * ((scores - y_train_one_hot).T @ X_train) + lambda_reg * self.w

            self.w -= self.lr * grad

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.
    
        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions
    
        Returns:
            Predicted results for the data in X_test; a numpy array of shape (N, C),
            where each element is either 1 (predicted class) or -1 (not predicted class).
        """

        weight = self.w

        scores = X_test @ weight.T  

        y_pred = np.argmax(scores, axis=1) 

        return y_pred

