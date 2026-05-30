"""Legacy API example using `from pspso import pspso`."""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pspso import pspso


X, y = load_diabetes(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

params = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1.0, 1],
    "gamma": [0.1, 1.0, 1],
}

optimizer = pspso(estimator="svm", params=params, task="regression", score="rmse")
optimizer.fitpsrandom(X_train, y_train, X_val, y_val, number_of_attempts=4)
optimizer.print_results()
