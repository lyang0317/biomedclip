"""
实验二：基于SVM的石油溢出（Oil Spill Classification）数据集识别
西南石油大学 机器学习实验
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示和美化样式（没起作用）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 魔幻写法70个等号做分割符号
print("=" * 70)
print("石油溢出数据集(Oil Spill Classification)分类识别实验")
print("=" * 70)

# ============================================================================
# 1. 数据加载与探索性分析
# ============================================================================
def load_oil_spill_data():
    """
    加载石油溢出数据集
    注意：实际数据集需要从UCI或Kaggle下载
    这里假设数据文件名为 'oil_spill.csv'
    """
    print("\n1. 数据加载与探索性分析")
    print("-" * 50)

    try:
        # 尝试加载数据集（这里需要替换为实际数据路径）
        # 数据可以从UCI获取：https://archive.ics.uci.edu/ml/datasets/Oil+Spill
        df = pd.read_csv('D:\cxdownload\oil_spill.csv')
        print(f"✓ 成功加载数据集，形状: {df.shape}")

    except FileNotFoundError:
        print("⚠ 未找到实际数据集文件，创建模拟数据用于演示")
        print("⚠ 实际实验中请使用真实石油溢出数据集")

    return df

# 加载数据
df = load_oil_spill_data()

# 显示数据基本信息
print(f"\n数据集信息:")
print(f"  样本数量: {df.shape[0]}")
print(f"  特征数量: {df.shape[1] - 1}")  # 减去目标列
print(f"  特征名称: {list(df.columns[:-1][:5])}...")  # 显示前5个特征

# 检查目标变量分布
target_counts = df['target'].value_counts()
print(f"\n目标变量分布:")
print(f"  非溢出 (0): {target_counts.get(0, 0)} 个样本 ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
print(f"  溢油 (1): {target_counts.get(1, 0)} 个样本 ({target_counts.get(1, 0)/len(df)*100:.1f}%)")

print(f"  类别不平衡比例: {target_counts.get(0, 1)/target_counts.get(1, 1):.1f}:1")

# ============================================================================
# 2. 数据预处理
# ============================================================================
print("\n\n2. 数据预处理")
print("-" * 50)

# 分离特征和标签
X = df.drop('target', axis=1).values
y = df['target'].values

# 检查缺失值
missing_values = np.isnan(X).sum()
print(f"缺失值数量: {missing_values}")

# 数据标准化（对SVM很重要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ 数据标准化完成")

# ============================================================================
# 3. 数据集划分
# ============================================================================
print("\n\n3. 数据集划分")
print("-" * 50)

# 方法1：常规训练-测试划分（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"方法1 - 训练-测试集划分 (7:3):")
print(f"  训练集: {X_train.shape[0]} 个样本 ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  测试集: {X_test.shape[0]} 个样本 ({X_test.shape[0]/len(X)*100:.1f}%)")

# 检查划分后的类别分布
train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)
print(f"\n训练集类别分布: 非溢出={train_counts[0]}, 溢油={train_counts[1]}")
print(f"测试集类别分布: 非溢出={test_counts[0]}, 溢油={test_counts[1]}")

# ============================================================================
# 4. 模型训练与评估
# ============================================================================
print("\n\n4. SVM模型训练与评估")
print("-" * 50)

def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel_type='linear', C=1.0):
    """
    训练和评估SVM模型
    """
    print(f"\n使用{kernel_type.upper()}核的SVM模型:")
    print(f"  参数: C={C}, kernel={kernel_type}")

    # 创建SVM模型
    if kernel_type == 'linear':
        clf = svm.SVC(kernel='linear', C=C, class_weight='balanced', random_state=42)
    elif kernel_type == 'rbf':
        clf = svm.SVC(kernel='rbf', C=C, gamma='scale', class_weight='balanced', random_state=42)
    else:
        clf = svm.SVC(kernel=kernel_type, C=C, class_weight='balanced', random_state=42)

    # 训练模型
    clf.fit(X_train, y_train)

    # 预测
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.decision_function(X_test)

    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 对于二分类问题，计算AUC
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_pred_prob)
    else:
        auc = None

    print(f"\n  评估结果:")
    print(f"    准确率 (Accuracy): {accuracy:.4f}")
    print(f"    精确率 (Precision): {precision:.4f}")
    print(f"    召回率 (Recall): {recall:.4f}")
    print(f"    F1分数: {f1:.4f}")
    if auc is not None:
        print(f"    AUC分数: {auc:.4f}")

    # 返回模型和评估结果
    return clf, y_pred, {'accuracy': accuracy, 'precision': precision,
                         'recall': recall, 'f1': f1, 'auc': auc}

# 4.1 使用线性SVM
print("\n4.1 线性SVM模型")
linear_clf, y_pred_linear, metrics_linear = train_and_evaluate_svm(
    X_train, X_test, y_train, y_test, kernel_type='linear', C=1.0
)

# 4.2 使用RBF核SVM
print("\n4.2 RBF核SVM模型")
rbf_clf, y_pred_rbf, metrics_rbf = train_and_evaluate_svm(
    X_train, X_test, y_train, y_test, kernel_type='rbf', C=1.0
)

# ============================================================================
# 5. 5折交叉验证
# ============================================================================
print("\n\n5. 5折交叉验证")
print("-" * 50)

# 使用分层K折交叉验证（对不平衡数据很重要）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 准备存储结果
cv_results = {'linear': [], 'rbf': []}

print("进行5折交叉验证...")

for train_idx, val_idx in skf.split(X_scaled, y):
    X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]

    # 线性SVM
    clf_linear_cv = svm.SVC(kernel='linear', C=1.0, class_weight='balanced')
    clf_linear_cv.fit(X_train_cv, y_train_cv)
    acc_linear = clf_linear_cv.score(X_val_cv, y_val_cv)
    cv_results['linear'].append(acc_linear)

    # RBF SVM
    clf_rbf_cv = svm.SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
    clf_rbf_cv.fit(X_train_cv, y_train_cv)
    acc_rbf = clf_rbf_cv.score(X_val_cv, y_val_cv)
    cv_results['rbf'].append(acc_rbf)

print("\n交叉验证结果:")
print(f"线性SVM - 各折准确率: {[f'{x:.4f}' for x in cv_results['linear']]}")
print(f"        平均准确率: {np.mean(cv_results['linear']):.4f} ± {np.std(cv_results['linear']):.4f}")

print(f"\nRBF核SVM - 各折准确率: {[f'{x:.4f}' for x in cv_results['rbf']]}")
print(f"        平均准确率: {np.mean(cv_results['rbf']):.4f} ± {np.std(cv_results['rbf']):.4f}")

# ============================================================================
# 6. 模型对比与详细分析
# ============================================================================
print("\n\n6. 模型对比与详细分析")
print("-" * 50)

# 6.1 生成详细分类报告
print("\n6.1 详细分类报告")

print("\n线性SVM分类报告:")
print(classification_report(y_test, y_pred_linear,
                            target_names=['非溢出', '溢油'],
                            digits=4))

print("\nRBF核SVM分类报告:")
print(classification_report(y_test, y_pred_rbf,
                            target_names=['非溢出', '溢油'],
                            digits=4))

# 6.2 混淆矩阵可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 线性SVM混淆矩阵
cm_linear = confusion_matrix(y_test, y_pred_linear)
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues',
            xticklabels=['no oil spill', 'oil spill'],
            yticklabels=['no oil spill', 'oil spill'],
            ax=axes[0])
axes[0].set_title('LINEAR SVM matrix')
axes[0].set_ylabel('real label')
axes[0].set_xlabel('predict label')

# RBF核SVM混淆矩阵
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['no oil spill', 'oil spill'],
            yticklabels=['no oil spill', 'oil spill'],
            ax=axes[1])
axes[1].set_title('RBF SVM matrix')
axes[1].set_ylabel('real label')
axes[1].set_xlabel('predict label')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ 混淆矩阵已保存为 'confusion_matrix.png'")
plt.show()

# 6.3 模型性能对比柱状图
metrics_df = pd.DataFrame({
    'LINEAR SVM': [metrics_linear['accuracy'], metrics_linear['precision'],
                metrics_linear['recall'], metrics_linear['f1']],
    'RBF SVM': [metrics_rbf['accuracy'], metrics_rbf['precision'],
                 metrics_rbf['recall'], metrics_rbf['f1']]
}, index=['Accuracy', 'Precision', 'Recall', 'F1 score'])

plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar', colormap='viridis', alpha=0.8)
plt.title('SVM performance comparison', fontsize=14)
plt.xlabel('assess index', fontsize=12)
plt.ylabel('score', fontsize=12)
plt.ylim(0, 1.1)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# 添加数值标签
for i, (idx, row) in enumerate(metrics_df.iterrows()):
    for j, val in enumerate(row):
        plt.text(i - 0.2 + j*0.4, val + 0.02, f'{val:.3f}',
                 ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 模型对比图已保存为 'model_comparison.png'")
plt.show()

# 6.4 交叉验证结果可视化
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(cv_results['linear']))

ax.plot(x_pos, cv_results['linear'], 'o-', linewidth=2, markersize=8,
        label='LINEAR SVM', color='blue')
ax.plot(x_pos, cv_results['rbf'], 's-', linewidth=2, markersize=8,
        label='RBF SVM', color='red')

# 添加平均值线
ax.axhline(y=np.mean(cv_results['linear']), color='blue', linestyle='--', alpha=0.5)
ax.axhline(y=np.mean(cv_results['rbf']), color='red', linestyle='--', alpha=0.5)

ax.set_xlabel('NUM', fontsize=12)
ax.set_ylabel('accuracy', fontsize=12)
ax.set_title('5 num cross validate result comparison', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'num {i+1}' for i in range(5)])
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_validation.png', dpi=300, bbox_inches='tight')
print("✓ 交叉验证图已保存为 'cross_validation.png'")
plt.show()

# ============================================================================
# 7. 特征重要性分析（仅适用于线性SVM）
# ============================================================================
print("\n\n7. 特征重要性分析")
print("-" * 50)

if hasattr(linear_clf, 'coef_'):
    # 获取特征重要性（权重）
    feature_importance = np.abs(linear_clf.coef_[0])

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': df.columns[:-1][:len(feature_importance)],
        '重要性': feature_importance
    })

    # 按重要性排序
    importance_df = importance_df.sort_values('重要性', ascending=False)

    print("\nTop 10 重要特征:")
    print(importance_df.head(10).to_string(index=False))

    # 可视化Top 20重要特征
    plt.figure(figsize=(12, 6))
    top_features = importance_df.head(20)
    plt.barh(range(len(top_features)), top_features['重要性'][::-1])
    plt.yticks(range(len(top_features)), top_features['特征'][::-1])
    plt.xlabel('feature importance（weight absolute）')
    plt.title('LINEAR SVM feature importance analyze (Top 20)')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ 特征重要性图已保存为 'feature_importance.png'")
    plt.show()
else:
    print("线性SVM特征重要性不可用（模型未收敛或非线性核）")

# ============================================================================
# 8. 实验结果总结
# ============================================================================
print("\n\n" + "=" * 70)
print("实验结果总结")
print("=" * 70)

print("\n1. 数据集分析:")
print(f"   - 总样本数: {len(df)}")
print(f"   - 特征数量: {X.shape[1]}")
print(f"   - 类别分布: 非溢出({target_counts.get(0, 0)}), 溢油({target_counts.get(1, 0)})")
print(f"   - 不平衡比例: {target_counts.get(0, 1)/target_counts.get(1, 1):.1f}:1")

print("\n2. 模型性能对比:")
print("   - 线性SVM: 准确率={:.4f}, F1={:.4f}".format(
    metrics_linear['accuracy'], metrics_linear['f1']))
print("   - RBF核SVM: 准确率={:.4f}, F1={:.4f}".format(
    metrics_rbf['accuracy'], metrics_rbf['f1']))

print("\n3. 交叉验证稳定性:")
print("   - 线性SVM: {:.4f} ± {:.4f}".format(
    np.mean(cv_results['linear']), np.std(cv_results['linear'])))
print("   - RBF核SVM: {:.4f} ± {:.4f}".format(
    np.mean(cv_results['rbf']), np.std(cv_results['rbf'])))

print("\n4. 结论与建议:")
if metrics_rbf['accuracy'] > metrics_linear['accuracy']:
    print("   - RBF核SVM在本数据集上表现略优于线性SVM")
    print("   - 可能原因是数据具有非线性可分特征")
else:
    print("   - 线性SVM在本数据集上表现良好")
    print("   - 可能原因是数据近似线性可分")

print("\n5. 改进建议:")
print("   - 针对不平衡数据，可使用SMOTE等技术进行过采样")
print("   - 可尝试调整SVM的C参数和gamma参数")
print("   - 可考虑使用集成学习方法如Random Forest")
print("   - 可进行特征选择，去除不相关特征")

# ============================================================================
# 9. 保存模型和结果
# ============================================================================
import joblib

# 保存最佳模型
joblib.dump(rbf_clf if metrics_rbf['f1'] > metrics_linear['f1'] else linear_clf,
            'best_oil_spill_model.pkl')
print("\n✓ 最佳模型已保存为 'best_oil_spill_model.pkl'")

# 保存预处理器
joblib.dump(scaler, 'scaler.pkl')
print("✓ 数据标准化器已保存为 'scaler.pkl'")

# 保存实验结果到文件
with open('experiment_results.txt', 'w', encoding='utf-8') as f:
    f.write("石油溢出数据集分类实验结果\n")
    f.write("="*50 + "\n\n")
    f.write(f"数据集: {df.shape}\n")
    f.write(f"训练集: {X_train.shape}, 测试集: {X_test.shape}\n\n")
    f.write("线性SVM结果:\n")
    for key, value in metrics_linear.items():
        f.write(f"  {key}: {value:.4f}\n")
    f.write("\nRBF核SVM结果:\n")
    for key, value in metrics_rbf.items():
        f.write(f"  {key}: {value:.4f}\n")

print("\n" + "=" * 70)
print("实验完成！所有结果已保存。")
print("=" * 70)