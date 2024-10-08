import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Generate or reuse synthetic data
# Generate synthetic data
x1 = np.arange(0, 10, 0.1)
x2 = np.arange(0, 10, 0.1)
x1, x2 = np.meshgrid(x1, x2)
y = np.sin(x1) * np.cos(x2) + np.random.normal(scale=0.1, size=x1.shape)

# Flatten the arrays
x1 = x1.flatten()
x2 = x2.flatten()
y = y.flatten()
X = np.vstack((x1, x2)).T

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
base_estimator=DecisionTreeRegressor(max_depth=5)
bagging_regressor = BaggingRegressor(base_estimator, n_estimators=50, random_state=42)
bagging_regressor.fit(X_train,y_train)

y_pred = bagging_regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2 =r2_score(y_test,y_pred)

print("Bagging regressor Mse:",mse)
print("Bagging regressor R2:",r2)