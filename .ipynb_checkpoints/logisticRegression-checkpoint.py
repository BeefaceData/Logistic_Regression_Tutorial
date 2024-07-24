"""
IMPLEMENTING LOGISTIC REGRESSION FROM SCRATCH WITH PYTHON
1. import libraries
2. create logistic regression class
3. define hyperparameter function
4. define sigmoid function
4. define compute loss function
5. define feed forward function
6. define fit function
7. define predict function
8. return prediction"""

import numpy as np


class LogisticRegression:
    # inittialising hyperparameters
    def __init__(self, learning_rate=0.0001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None 
        self.losses = []

    # define sigmoid function
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    # define compute loss
    def compute_loss(self, y_true, y_pred):
        # binary entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    # define feed forward
    def feed_forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(z)
        return A

    # define fit
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initiate parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            A = self.feed_forward(X)
            self.losses.append(self.compute_loss(y,A))
            dz = A - y

            # compute gradient
            dw = (1/n_samples) * np.dot(X.T, dz)
            db = (1/n_samples) * np.sum(dz)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # define predict
    def predict(self, X):
        threshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]

        return y_predicted, np.array(y_predicted_cls)
    
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

X, y = dataset.data, dataset.target 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
regressor.fit(X_train, y_train)
y_pred, y_pred_cls = regressor.predict(X_test)
cm = confusion_matrix(y_test, y_pred_cls)
accuracy = accuracy_score(y_test, y_pred_cls)
print("Test accuracy: {0:.3f}".format(accuracy))
print("Confusion Matrix:", cm)