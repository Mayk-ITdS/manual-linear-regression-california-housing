import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from ETL.transform import gauss_jordan_reverse

class Solver(ABC):
    @abstractmethod
    def fit(self, x, y, model):
        pass

class LinearRegression:
    def __init__(self, solver: "Solver"):
        self.solver = solver
        self.model = RegressionModel()

    def fit(self, x, y):
        self.solver.fit(x, y, self.model)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        y_pred = self.predict(x)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot

class RegressionModel:
    def __init__(self):
        self.w = None
        self.b = None

    def predict(self, x):
        return x @ self.w + self.b

class NormalSolver(Solver):
    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def fit(self, x, y, model):
        n, p = x.shape
        a = x.T @ x
        if self.lambda_ > 0:
            a += self.lambda_ * np.eye(p)
        a_inv = gauss_jordan_reverse(a)
        model.w = a_inv @ (x.T @ y)
        model.b = y.mean() - x.mean(axis=0) @ model.w

class GradientSolver(Solver):
    def __init__(self, learning_rate=0.1, epochs=1000, ridge_lambda=0.0, patience=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.ridge_lambda = ridge_lambda
        self.patience = patience
        self.history = {"mse": [], "r2": [], "bias": [], "weights": []}
        self.best_model = None

    def fit(self, x, y, model):
        n, p = x.shape
        model.w = np.zeros(p)
        model.b = 0.0
        best_mse = float("inf")
        no_improvement = 0
        lr = self.learning_rate
        for epoch in range(self.epochs):
            assert isinstance(model.w, np.ndarray), f"model.w changed type to {type(model.w)}"
            y_pred = x @ model.w + model.b
            e = y - y_pred
            mse = (e.T @ e) / n

            grad_w = -(1/n) * (x.T @ e)
            grad_b = -(1/n) * e.sum()

            if self.ridge_lambda > 1e-12:
                grad_w += (self.ridge_lambda/n) * model.w

            max_val = 5.0
            grad_w = np.clip(grad_w, -max_val, max_val)

            model.w -= lr * grad_w
            model.b -= lr * grad_b

            if (epoch + 1) % 1000 == 0:
                    lr *= 0.9

            r2 = 1 - (e.T @ e) / np.sum((y - y.mean())**2)
            self.history["mse"].append(float(mse))
            self.history["r2"].append(float(r2))
            self.history["bias"].append(float(model.b))
            self.history["weights"].append(model.w.copy())

            if mse < best_mse:
                best_mse = mse
                self.best_model = {
                    "weights": model.w.copy(),
                    "bias": model.b,
                    "mse": float(mse),
                    "r2": float(r2)
                }
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, MSE={mse:.4f}, R2={r2:.4f}")

    def save_best_model(self, filepath="best_model.csv"):
        if self.best_model is None:
            raise ValueError("Best model not found.")

        row = {
            "bias": self.best_model["bias"],
            "mse": self.best_model["mse"],
            "r2": self.best_model["r2"],
        }

        row.update({f"w{i}": float(w) for i, w in enumerate(self.best_model["weights"])})

        df = pd.DataFrame([row])
        df.to_csv(filepath, index=False)
        print(f"Best model saved to {filepath}")

    def save_training_history(self, filepath="training_history.csv"):
        results = []
        for epoch,mse,r2,bias,weights in zip(range(len(self.history["mse"])),
                                             self.history["mse"],
                                             self.history["r2"],
                                             self.history["bias"],
                                             self.history["weights"]
                                             ):
            row = {
                "epoch": epoch,
                "mse": mse,
                "r2": r2,
                "bias": bias,
            }
            for i,w in enumerate(weights):
                row[f"w{i}"] = float(w)
            results.append(row)

        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        print(f"Training history saved to {filepath}")
