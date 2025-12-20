import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1) 加载糖尿病数据集，观察数据
diabetes = load_diabetes()
print("数据集特征名称:", diabetes.feature_names)
print("数据形状:", diabetes.data.shape)
print("目标变量形状:", diabetes.target.shape)

# 将数据组合成DataFrame
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target
print("\n数据集前5行:")
print(df.head())
print("\n数据集基本信息:")
print(df.info())

# 2) 基于线性回归对数据集进行分析
X = df.drop('target', axis=1)
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")

# 建立线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n多变量线性回归模型评估:")
print(f"均方误差(MSE): {mse:.4f}")
print(f"决定系数(R²): {r2:.4f}")
print("模型系数:", model.coef_)
print("模型截距:", model.intercept_)

# 3) 考察每个特征值与结果之间的关联性
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    ax = axes[i]
    ax.scatter(X[col], y, alpha=0.5, s=10)
    ax.set_xlabel(col)
    ax.set_ylabel('target')
    ax.set_title(f'{col} vs target')
    # 计算相关系数
    corr = np.corrcoef(X[col], y)[0, 1]
    ax.text(0.05, 0.95, f'corr={corr:.3f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# 找出最相关的特征
correlations = []
for col in X.columns:
    corr = np.corrcoef(X[col], y)[0, 1]
    correlations.append((col, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)
print("\n特征与目标变量的相关性排序（按绝对值）:")
for col, corr in correlations:
    print(f"{col}: {corr:.4f}")

most_corr_feature = correlations[0][0]
print(f"\n最相关的特征: {most_corr_feature}")

# 4) 使用最相关特征进行单变量回归分析
X_single = df[[most_corr_feature]]
y_single = df['target']

# 划分数据集
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y_single, test_size=0.2, random_state=42)

# 创建线性回归模型
model_single = LinearRegression()
model_single.fit(X_train_s, y_train_s)

# 预测
y_pred_s = model_single.predict(X_test_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print(f"\n单变量线性回归（使用特征 '{most_corr_feature}'）:")
print(f"均方误差(MSE): {mse_s:.4f}")
print(f"决定系数(R²): {r2_s:.4f}")
print(f"权重系数: {model_single.coef_[0]:.4f}")
print(f"截距: {model_single.intercept_:.4f}")

# 可视化单变量回归结果
plt.figure(figsize=(10, 6))
# 训练集散点
plt.scatter(X_train_s, y_train_s, color='blue', alpha=0.5, label='Train data', s=20)
# 测试集散点
plt.scatter(X_test_s, y_test_s, color='green', alpha=0.5, label='Test data', s=20)
# 回归线
x_range = np.linspace(X_single.min(), X_single.max(), 100).reshape(-1, 1)
y_range_pred = model_single.predict(x_range)
plt.plot(x_range, y_range_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel(most_corr_feature)
plt.ylabel('Diabetes progression')
plt.title(f'Linear Regression: {most_corr_feature} vs Diabetes progression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 预测值与真实值对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test_s, y_pred_s, alpha=0.7, s=50)
plt.plot([y_test_s.min(), y_test_s.max()], [y_test_s.min(), y_test_s.max()],
         'r--', lw=2, label='Ideal prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'True vs Predicted (Feature: {most_corr_feature})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 输出模型预测示例
print("\n预测示例（前5个测试样本）:")
for i in range(5):
    print(f"真实值: {y_test_s.iloc[i]:.2f}, 预测值: {y_pred_s[i]:.2f}, 误差: {abs(y_test_s.iloc[i] - y_pred_s[i]):.2f}")