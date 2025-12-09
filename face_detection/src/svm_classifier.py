import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0):
        self.kernel = kernel
        self.C = C
        self.scaler = StandardScaler()
        self.svm = SVC(kernel=kernel, C=C, gamma='scale', probability=True, class_weight='balanced')
        self.is_trained = False

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained!")
        X_scaled = self.scaler.transform(X)
        return self.svm.predict(X_scaled)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained!")
        X_scaled = self.scaler.transform(X)
        return self.svm.predict_proba(X_scaled)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=['Non-face', 'Face'])
        return accuracy, report
