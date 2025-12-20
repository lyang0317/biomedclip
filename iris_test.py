import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集（7:3 比例）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 线性 SVM
svm_linear = svm.SVC(kernel='linear', C=1.0, decision_function_shape='ovr')

# 核化 SVM（使用 RBF 核）
svm_rbf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr')

svm_linear.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# 预测
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)

# 输出分类报告
print("线性 SVM 分类报告：")
print(classification_report(y_test, y_pred_linear))

print("RBF 核 SVM 分类报告：")
print(classification_report(y_test, y_pred_rbf))

# 混淆矩阵
print("线性 SVM 混淆矩阵：")
print(confusion_matrix(y_test, y_pred_linear))

print("RBF 核 SVM 混淆矩阵：")
print(confusion_matrix(y_test, y_pred_rbf))

# 使用交叉验证评估模型
scores_linear = cross_val_score(svm_linear, X, y, cv=5, scoring='accuracy')
scores_rbf = cross_val_score(svm_rbf, X, y, cv=5, scoring='accuracy')

print("线性 SVM 5折交叉验证准确率：", scores_linear.mean())
print("RBF 核 SVM 5折交叉验证准确率：", scores_rbf.mean())

# 以两个特征为例，可视化决策边界（仅适用于 2D 特征）
def plot_decision_boundary(clf, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()

# 仅使用前两个特征进行可视化
X_2d = iris.data[:, :2]
clf = svm.SVC(kernel='linear').fit(X_2d, y)
plot_decision_boundary(clf, X_2d, y, "SVM with Linear Kernel")