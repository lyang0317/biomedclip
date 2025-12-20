import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
train_df = pd.read_csv('D:\cxdownload\\train.csv')
test_df = pd.read_excel('D:\cxdownload\\test.xlsx')

print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)
print("\n训练集列名:")
print(train_df.columns.tolist())
print("\n训练集前5行:")
print(train_df.head())
print("\n训练集基本信息:")
print(train_df.info())
print("\n训练集描述性统计:")
print(train_df.describe())



# 检查缺失值
missing_train = train_df.isnull().sum()
missing_test = test_df.isnull().sum()

print("训练集缺失值统计:")
print(missing_train[missing_train > 0])
print(f"训l练集总缺失值：{missing_train.sum()}")
print("\n测试集缺失值统计:")
print(missing_test[missing_test > 0])
print(f"测试集总缺失值:{missing_test.sum()}")


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
missing_train_nonzero =missing_train[missing_train >0]
if not missing_train_nonzero.empty:
    missing_train[missing_train > 0].plot(kind='bar', color='skyblue')
    plt.title('训练集缺失值情况')
    plt.xlabel('特征')
    plt.ylabel('缺失值数量')
else:
    plt.text(0.5, 0.5, '无缺失值', ha='center', va='center', fontsize=12)
    plt.title('训练集缺失值情况 - 无缺失值')

plt.subplot(1, 2, 2)
missing_test_nonzero = missing_test[missing_test > 0]
if not missing_test_nonzero.empty:
    missing_test[missing_test > 0].plot(kind='bar', color='lightcoral')
    plt.title('测试集缺失值情况')
    plt.xlabel('特征')
    plt.ylabel('缺失值数量')
else:
    plt.text(0.5, 0.5, '无缺失值', ha='center', va='center', fontsize=12)
    plt.title('测试集缺失值情况 - 无缺失值')

plt.tight_layout()
plt.savefig('missing_values.png', dpi=300)
plt.show()

print("\n训练集缺失值总数:", missing_train.sum())
print("测试集缺失值总数:", missing_test.sum())

# 处理缺失值（如果存在）
if missing_train.sum() > 0:
    train_df = train_df.fillna(train_df.median())
if missing_test.sum() > 0:
    test_df = test_df.fillna(test_df.median())




# 标签分布
plt.figure(figsize=(10, 6))
label_counts = train_df['fetal_health'].value_counts()
colors = ['lightgreen', 'orange', 'red']
labels = ['正常 (1)', '疑似 (2)', '病理 (3)']
plt.bar(labels, label_counts.values, color=colors, alpha=0.8)
plt.title('胎儿健康类别分布')
plt.xlabel('健康类别')
plt.ylabel('样本数量')
for i, v in enumerate(label_counts.values):
    plt.text(i, v + 10, str(v), ha='center', va='bottom')
plt.savefig('label_distribution.png', dpi=300)
plt.show()

print("类别分布:")
print(label_counts)
print("\n类别比例:")
print(label_counts / len(train_df))

# 数值特征分布
numeric_features = train_df.select_dtypes(include=[np.number]).columns
numeric_features = numeric_features.drop('fetal_health')

fig, axes = plt.subplots(5, 4, figsize=(20, 20))
axes = axes.flatten()

for i, col in enumerate(numeric_features[:20]):
    axes[i].hist(train_df[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[i].set_title(f'{col}分布')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('频数')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300)
plt.show()

# 箱线图检测异常值
plt.figure(figsize=(15, 10))
train_df[numeric_features[:10]].boxplot()
plt.title('前10个特征的箱线图')
plt.xticks(rotation=45)
plt.savefig('boxplot_outliers.png', dpi=300)
plt.show()




# 计算相关性矩阵
correlation_matrix = train_df.corr()

# 目标变量的相关性
target_correlation = correlation_matrix['fetal_health'].sort_values(ascending=False)

plt.figure(figsize=(12, 8))
target_correlation.drop('fetal_health').plot(kind='bar', color='teal', alpha=0.7)
plt.title('特征与目标变量的相关性')
plt.xlabel('特征')
plt.ylabel('相关系数')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('target_correlation.png', dpi=300)
plt.show()

# 热力图
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
            square=True, cbar_kws={"shrink": 0.8})
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

# 选择与目标变量相关性高的特征
high_corr_features = target_correlation[abs(target_correlation) > 0.1].index.tolist()
high_corr_features.remove('fetal_health')
print("\n")
print(f"选择{len(high_corr_features)}个高相关性特征:")
print(high_corr_features)




# 准备特征和标签
X = train_df.drop('fetal_health', axis=1)
y = train_df['fetal_health']

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

# 特征选择方法1：方差分析（ANOVA）
selector = SelectKBest(f_classif, k=15)
X_selected = selector.fit_transform(X_scaled, y)
selected_features_idx = selector.get_support(indices=True)
selected_features = X.columns[selected_features_idx]

print(f"选择的特征 ({len(selected_features)}个):")
print(selected_features.tolist())

# 特征选择方法2：PCA降维
pca = PCA(n_components=0.95)  # 保留95%的方差
X_pca = pca.fit_transform(X_scaled)
test_pca = pca.transform(test_scaled)

print(f"PCA降维后维度: {X_pca.shape[1]}")
print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.3f}")

# 可视化PCA解释方差
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.title('PCA累积解释方差')
plt.xlabel('主成分数量')
plt.ylabel('累积解释方差比例')
plt.grid(True, alpha=0.3)
plt.savefig('pca_variance.png', dpi=300)
plt.show()



# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 定义模型
models = {
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 训练并评估模型
results = {}
for name, model in models.items():
    print(f"\n训练{name}模型...")
    model.fit(X_train, y_train)

    # 训练集准确率
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # 验证集准确率
    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'val_f1': val_f1
    }

    print(f"{name} - 训练集准确率: {train_acc:.4f}")
    print(f"{name} - 验证集准确率: {val_acc:.4f}")
    print(f"{name} - 验证集F1分数: {val_f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_val, val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'{name}混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(f'confusion_matrix_{name}.png', dpi=300)
    plt.show()

# 交叉验证
print("\n交叉验证结果:")
for name in models.keys():
    cv_scores = cross_val_score(models[name], X_scaled, y, cv=5, scoring='accuracy')
    print(f"{name} - 交叉验证平均准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 模型性能对比
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
val_accuracies = [results[name]['val_accuracy'] for name in model_names]
val_f1_scores = [results[name]['val_f1'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, val_accuracies, width, label='准确率', color='skyblue')
rects2 = ax.bar(x + width/2, val_f1_scores, width, label='F1分数', color='lightcoral')

ax.set_xlabel('模型')
ax.set_ylabel('分数')
ax.set_title('模型性能对比')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.set_ylim([0, 1])

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()





# SVM参数调优
print("\nSVM参数调优...")
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_svm = GridSearchCV(SVC(random_state=42), param_grid_svm,
                        cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_svm.fit(X_train, y_train)

print(f"最佳参数: {grid_svm.best_params_}")
print(f"最佳分数: {grid_svm.best_score_:.4f}")

# 随机森林参数调优
print("\n随机森林参数调优...")
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf,
                       cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

print(f"最佳参数: {grid_rf.best_params_}")
print(f"最佳分数: {grid_rf.best_score_:.4f}")

# 使用最佳模型
best_model = grid_rf.best_estimator_ if grid_rf.best_score_ > grid_svm.best_score_ else grid_svm.best_estimator_
best_model_name = 'Random Forest' if grid_rf.best_score_ > grid_svm.best_score_ else 'SVM'
print(f"\n选择最佳模型: {best_model_name}")







# 使用最佳模型进行预测
final_model = best_model
final_model.fit(X_scaled, y)  # 使用全部训练数据重新训练

# 对测试集进行预测
test_predictions = final_model.predict(test_scaled)

# 保存预测结果
predict_df = pd.DataFrame({
    'fetal_health': test_predictions
})

predict_df.to_excel('D:\cxdownload\predict.xlsx', index=False)
print("预测结果已保存到 predict.csv")
print(f"预测结果分布:")
print(predict_df['fetal_health'].value_counts().sort_index())

# 可视化预测结果分布
plt.figure(figsize=(8, 6))
predict_counts = predict_df['fetal_health'].value_counts().sort_index()
colors = ['lightgreen', 'orange', 'red']
labels = ['正常 (1)', '疑似 (2)', '病理 (3)']

plt.bar(labels, predict_counts.values, color=colors, alpha=0.8)
plt.title('测试集预测结果分布')
plt.xlabel('预测类别')
plt.ylabel('样本数量')
for i, v in enumerate(predict_counts.values):
    plt.text(i, v + 2, str(v), ha='center', va='bottom')
plt.savefig('prediction_distribution.png', dpi=300)
plt.show()




if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'].values,
             color='teal', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('特征重要性')
    plt.title('Top 15 特征重要性')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()