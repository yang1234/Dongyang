import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression

# 1. 加载数据
data = fetch_california_housing()
X = data.data
y = data.target

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 探索不同的树深度
depths = range(1, 21)
mse_scores = {'Decision Tree': [], 'Random Forest': [], 'AdaBoost': []}
r2_scores = {'Decision Tree': [], 'Random Forest': [], 'AdaBoost': []}

for depth in depths:
    # 更新决策树和 AdaBoost 的基估计器
    tree = DecisionTreeRegressor(max_depth=depth)
    
    # 4. 初始化模型（除去线性回归）
    models = {
        "Decision Tree": DecisionTreeRegressor(max_depth=depth),
        "Random Forest": RandomForestRegressor(n_estimators=40, max_depth=depth, random_state=42),
        "AdaBoost": AdaBoostRegressor(tree, n_estimators=40, random_state=42)
    }

    # 5. 训练和评估模型
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores[name].append(mse)
        r2_scores[name].append(r2)

# 6. 可视化不同深度的模型性能
plt.figure(figsize=(12, 6))
for name, scores in mse_scores.items():
    plt.plot(depths, scores, label=f'{name} MSE')
plt.xlabel('Depth of Trees')
plt.ylabel('Mean Squared Error')
plt.title('MSE by Tree Depth for Various Models')
plt.legend()
plt.grid(True)
plt.show()

# 7. 输出每个模型的最佳深度和MSE
for name, scores in mse_scores.items():
    best_depth = np.argmin(scores) + 1  # +1 because depth range starts at 1
    best_r2 = max(r2_scores[name])
    best_r2_depth = r2_scores[name].index(best_r2) + 1  
    print(f"Best Depth for {name}: {best_depth}, Minimum MSE: {min(scores)}, Best R²: {best_r2} at depth {best_r2_depth}")
