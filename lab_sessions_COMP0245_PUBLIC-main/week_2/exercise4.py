import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
base_estimator=DecisionTreeRegressor(max_depth=4)
# Initialize AdaBoostRegressor with a base DecisionTreeRegressor
ada_regressor = AdaBoostRegressor(base_estimator, n_estimators=50, random_state=42, loss='linear')

# Fit the model on the training data
ada_regressor.fit(X_train, y_train)

# Predict on the training data to compute prediction error
y_train_pred = ada_regressor.predict(X_train)

# Calculate MSE and R^2 for the training set predictions
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

# Print performance metrics
print("AdaBoost Regressor Training MSE:", mse_train)
print("AdaBoost Regressor Training R^2:", r2_train)

# Optionally plot the training data and model predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, color='blue', label='Actual', alpha=0.5)
plt.scatter(X_train[:, 0], y_train_pred, color='red', label='Predicted', alpha=0.5)
plt.title('Comparison of Actual and Predicted Values on Training Data')
plt.xlabel('X1')
plt.ylabel('Y')
plt.legend()
plt.show()
