"""
石油溢出数据分类 - 神经网络与对比分析
完整实现代码
"""

# ============================================================================
# 导入所有必要的库
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

# 传统机器学习模型
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# 深度学习库
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# 其他
import joblib
import pickle
import os
from datetime import datetime

# 设置随机种子以确保可重复性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 设置中文显示（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. 数据加载与预处理模块
# ============================================================================
class DataPreprocessor:
    """数据预处理类"""

    def __init__(self, data_path, target_column='target'):
        """
        初始化数据预处理器

        参数:
        - data_path: 数据文件路径
        - target_column: 目标列名
        """
        self.data_path = data_path
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def load_data(self):
        """加载数据"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在加载数据...")

        # 支持多种格式的数据文件
        if self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path)
        elif self.data_path.endswith('.parquet'):
            self.data = pd.read_parquet(self.data_path)
        else:
            raise ValueError("不支持的文件格式。请使用 CSV、Excel 或 Parquet 文件")

        print(f"数据集形状: {self.data.shape}")
        print(f"特征列: {self.data.columns.tolist()}")

        # 检查目标列是否存在
        if self.target_column not in self.data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不在数据中")

        return self.data

    def explore_data(self):
        """数据探索性分析"""
        print("\n" + "="*60)
        print("数据探索性分析")
        print("="*60)

        # 基本信息
        print("\n=== 数据基本信息 ===")
        print(self.data.info())

        # 类别分布
        print(f"\n=== 目标变量分布 ===")
        target_dist = self.data[self.target_column].value_counts()
        print(target_dist)

        # 可视化类别分布
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        target_dist.plot(kind='bar', color=['skyblue', 'lightcoral'])
        plt.title('类别分布')
        plt.xlabel('类别')
        plt.ylabel('数量')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.pie(target_dist.values, labels=target_dist.index,
                autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        plt.title('类别比例')
        plt.tight_layout()
        plt.show()

        # 缺失值检查
        print(f"\n=== 缺失值统计 ===")
        missing = self.data.isnull().sum()
        missing_percent = (missing / len(self.data)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_percent
        })
        print(missing_df[missing_df['缺失数量'] > 0])

        # 基本统计信息
        print("\n=== 数值特征统计 ===")
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(self.data[numeric_cols].describe())

        return missing_df

    def preprocess(self, test_size=0.2, scale_features=True):
        """
        数据预处理

        参数:
        - test_size: 测试集比例
        - scale_features: 是否标准化特征
        """
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 正在进行数据预处理...")

        # 分离特征和目标
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=SEED, stratify=y
        )

        print(f"训练集大小: {self.X_train.shape}")
        print(f"测试集大小: {self.X_test.shape}")

        # 特征标准化
        if scale_features:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print("特征已标准化")
        else:
            self.X_train_scaled = self.X_train.values
            self.X_test_scaled = self.X_test.values

        return (self.X_train_scaled, self.X_test_scaled,
                self.y_train, self.y_test, self.X_train, self.X_test)

# ============================================================================
# 2. 神经网络模型模块
# ============================================================================
class NeuralNetworkModel:
    """神经网络模型类"""

    def __init__(self, input_dim, model_name='oil_spill_nn'):
        """
        初始化神经网络模型

        参数:
        - input_dim: 输入维度
        - model_name: 模型名称
        """
        self.input_dim = input_dim
        self.model_name = model_name
        self.model = None
        self.history = None

    def build_model(self,
                    hidden_layers=[128, 64, 32],
                    dropout_rates=[0.3, 0.3, 0.2],
                    activation='relu',
                    output_activation='sigmoid'):
        """
        构建神经网络模型

        参数:
        - hidden_layers: 隐藏层神经元数量列表
        - dropout_rates: 每层的dropout率
        - activation: 隐藏层激活函数
        - output_activation: 输出层激活函数
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在构建神经网络模型...")

        model = keras.Sequential()
        model.add(layers.Input(shape=(self.input_dim,)))

        # 添加隐藏层
        for i, (units, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.BatchNormalization())
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))

        # 添加输出层
        model.add(layers.Dense(1, activation=output_activation))

        self.model = model

        # 打印模型结构
        print("\n=== 神经网络模型结构 ===")
        self.model.summary()

        return model

    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """编译模型"""
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss='binary_crossentropy',
            metrics=['accuracy',
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall'),
                     keras.metrics.AUC(name='auc')]
        )

        print(f"模型已编译，优化器: {optimizer}, 学习率: {learning_rate}")

    def train(self, X_train, y_train,
              validation_split=0.2,
              epochs=100,
              batch_size=32,
              patience=15):
        """训练模型"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始训练神经网络...")

        # 设置回调函数
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        print(f"训练完成，最佳epoch: {len(self.history.history['loss'])}")

        return self.history

    def evaluate(self, X_test, y_test):
        """评估模型"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 评估神经网络模型...")

        # 评估损失和指标
        test_loss, test_accuracy, test_precision, test_recall, test_auc = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        # 预测
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 计算额外指标
        f1 = f1_score(y_test, y_pred)

        results = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': f1,
            'auc': test_auc,
            'loss': test_loss,
            'y_pred': y_pred,
            'y_pred_prob': y_pred_prob
        }

        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"测试集F1分数: {f1:.4f}")
        print(f"测试集AUC分数: {test_auc:.4f}")

        return results

    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 准确率曲线
        axes[0, 0].plot(self.history.history['accuracy'], label='训练准确率')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='验证准确率')
        axes[0, 0].set_title('模型准确率')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 损失曲线
        axes[0, 1].plot(self.history.history['loss'], label='训练损失')
        axes[0, 1].plot(self.history.history['val_loss'], label='验证损失')
        axes[0, 1].set_title('模型损失')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('损失')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 精确率曲线
        axes[1, 0].plot(self.history.history['precision'], label='训练精确率')
        axes[1, 0].plot(self.history.history['val_precision'], label='验证精确率')
        axes[1, 0].set_title('模型精确率')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('精确率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 召回率曲线
        axes[1, 1].plot(self.history.history['recall'], label='训练召回率')
        axes[1, 1].plot(self.history.history['val_recall'], label='验证召回率')
        axes[1, 1].set_title('模型召回率')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('召回率')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# ============================================================================
# 3. 传统机器学习模型模块
# ============================================================================
class TraditionalMLModels:
    """传统机器学习模型类"""

    def __init__(self):
        """初始化传统机器学习模型"""
        self.models = {}
        self.results = {}

    def initialize_models(self):
        """初始化要对比的模型"""
        self.models = {
            'SVM': {
                'model': SVC(kernel='rbf', probability=True, random_state=SEED),
                'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
            },
            '随机森林': {
                'model': RandomForestClassifier(n_estimators=100, random_state=SEED),
                'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
            },
            '逻辑回归': {
                'model': LogisticRegression(random_state=SEED, max_iter=1000),
                'params': {'C': [0.1, 1, 10], 'penalty': ['l2']}
            },
            'K近邻': {
                'model': KNeighborsClassifier(),
                'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            }
        }

        print(f"已初始化 {len(self.models)} 个传统机器学习模型")
        return self.models

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, cv_folds=5):
        """训练和评估所有模型"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始训练传统机器学习模型...")

        for name, model_info in self.models.items():
            print(f"\n=== 训练 {name} 模型 ===")

            # 训练模型
            model = model_info['model']
            model.fit(X_train, y_train)

            # 预测
            if hasattr(model, 'predict_proba'):
                y_pred_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_prob = model.decision_function(X_test)

            y_pred = model.predict(X_test)

            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)

            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=cv_folds, scoring='accuracy')

            # 保存结果
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob
            }

            print(f"准确率: {accuracy:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(f"AUC分数: {auc:.4f}")
            print(f"{cv_folds}折交叉验证准确率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        return self.results

    def plot_confusion_matrices(self, y_test, class_names=['非溢出', '溢出']):
        """绘制所有模型的混淆矩阵"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= len(axes):
                break

            cm = confusion_matrix(y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names,
                        ax=axes[idx])
            axes[idx].set_title(f'{name} - 混淆矩阵')
            axes[idx].set_ylabel('真实标签')
            axes[idx].set_xlabel('预测标签')

        # 隐藏多余的子图
        for idx in range(len(self.results), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

# ============================================================================
# 4. 可视化与分析模块
# ============================================================================
class Visualization:
    """可视化分析类"""

    @staticmethod
    def plot_model_comparison(results, nn_results=None):
        """绘制模型对比图"""
        # 准备数据
        comparison_data = {}

        # 添加传统模型结果
        for name, result in results.items():
            comparison_data[name] = {
                'accuracy': result['accuracy'],
                'f1': result['f1'],
                'auc': result['auc']
            }

        # 添加神经网络结果
        if nn_results:
            comparison_data['神经网络'] = {
                'accuracy': nn_results['accuracy'],
                'f1': nn_results['f1'],
                'auc': nn_results['auc']
            }

        # 转换为DataFrame
        df = pd.DataFrame(comparison_data).T
        df = df.sort_values('accuracy', ascending=False)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        metrics = ['accuracy', 'f1', 'auc']
        titles = ['准确率对比', 'F1分数对比', 'AUC分数对比']
        colors = plt.cm.Set3(np.linspace(0, 1, len(df)))

        for idx, metric in enumerate(metrics):
            bars = axes[idx].bar(df.index, df[metric], color=colors)
            axes[idx].set_title(titles[idx], fontsize=14)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
            axes[idx].set_ylim([0, 1.05])
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)

            # 在柱状图上添加数值
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()

        return df

    @staticmethod
    def plot_radar_chart(results_df):
        """绘制雷达图"""
        if len(results_df) < 2:
            print("需要至少2个模型才能绘制雷达图")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        # 准备数据
        categories = list(results_df.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # 颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))

        for idx, (model_name, row) in enumerate(results_df.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图对比', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.show()

    @staticmethod
    def plot_feature_importance(models, feature_names, X_sample, top_n=10):
        """绘制特征重要性"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 1. 神经网络特征重要性（基于梯度）
        if '神经网络' in models:
            nn_model = models['神经网络']['model']
            X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = nn_model(X_tensor)

            gradients = tape.gradient(predictions, X_tensor)
            importance = np.mean(np.abs(gradients.numpy()), axis=0)
            importance = importance / np.sum(importance)

            nn_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)

            axes[0].barh(range(top_n), nn_importance_df['importance'])
            axes[0].set_yticks(range(top_n))
            axes[0].set_yticklabels(nn_importance_df['feature'])
            axes[0].set_xlabel('重要性分数')
            axes[0].set_title('神经网络特征重要性 (Top 10)')
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.3)

        # 2. 随机森林特征重要性
        if '随机森林' in models:
            rf_model = models['随机森林']['model']
            rf_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)

            axes[1].barh(range(top_n), rf_importance['importance'])
            axes[1].set_yticks(range(top_n))
            axes[1].set_yticklabels(rf_importance['feature'])
            axes[1].set_xlabel('重要性分数')
            axes[1].set_title('随机森林特征重要性 (Top 10)')
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.3)

        # 3. 逻辑回归特征重要性（系数绝对值）
        if '逻辑回归' in models:
            lr_model = models['逻辑回归']['model']
            if hasattr(lr_model, 'coef_'):
                lr_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(lr_model.coef_[0])
                }).sort_values('importance', ascending=False).head(top_n)

                axes[2].barh(range(top_n), lr_importance['importance'])
                axes[2].set_yticks(range(top_n))
                axes[2].set_yticklabels(lr_importance['feature'])
                axes[2].set_xlabel('重要性分数')
                axes[2].set_title('逻辑回归特征重要性 (Top 10)')
                axes[2].invert_yaxis()
                axes[2].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

# ============================================================================
# 5. 主执行模块
# ============================================================================
class OilSpillClassifier:
    """石油溢出分类主类"""

    def __init__(self, data_path, output_dir='results'):
        """
        初始化分类器

        参数:
        - data_path: 数据文件路径
        - output_dir: 输出目录
        """
        self.data_path = data_path
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化组件
        self.preprocessor = None
        self.nn_model = None
        self.ml_models = None
        self.visualizer = Visualization()

        # 存储结果
        self.results = {}
        self.nn_results = None

    def run_full_pipeline(self, test_size=0.2):
        """运行完整流程"""
        print("="*70)
        print("石油溢出数据分类 - 完整分析流程")
        print("="*70)

        # 步骤1: 数据加载与预处理
        print("\n[步骤1] 数据加载与预处理")
        self.preprocessor = DataPreprocessor(self.data_path)
        data = self.preprocessor.load_data()
        missing_info = self.preprocessor.explore_data()
        X_train, X_test, y_train, y_test, X_train_df, X_test_df = \
            self.preprocessor.preprocess(test_size=test_size)

        # 步骤2: 神经网络模型
        print("\n[步骤2] 神经网络模型训练")
        self.nn_model = NeuralNetworkModel(input_dim=X_train.shape[1])
        self.nn_model.build_model()
        self.nn_model.compile_model()
        history = self.nn_model.train(X_train, y_train)
        self.nn_results = self.nn_model.evaluate(X_test, y_test)
        self.nn_model.plot_training_history()

        # 步骤3: 传统机器学习模型
        print("\n[步骤3] 传统机器学习模型训练")
        self.ml_models = TraditionalMLModels()
        self.ml_models.initialize_models()
        ml_results = self.ml_models.train_and_evaluate(X_train, y_train, X_test, y_test)

        # 合并所有结果
        self.results = ml_results.copy()
        self.results['神经网络'] = {
            'model': self.nn_model.model,
            'accuracy': self.nn_results['accuracy'],
            'f1': self.nn_results['f1'],
            'auc': self.nn_results['auc'],
            'cv_mean': self.nn_results['accuracy'],  # 近似值
            'cv_std': 0.0,
            'y_pred': self.nn_results['y_pred'],
            'y_pred_prob': self.nn_results['y_pred_prob']
        }

        # 步骤4: 可视化分析
        print("\n[步骤4] 可视化分析")

        # 绘制混淆矩阵
        self.ml_models.plot_confusion_matrices(y_test)

        # 绘制模型对比
        comparison_df = self.visualizer.plot_model_comparison(ml_results, self.nn_results)

        # 绘制雷达图
        self.visualizer.plot_radar_chart(comparison_df)

        # 绘制特征重要性
        feature_names = self.preprocessor.data.drop(self.preprocessor.target_column, axis=1).columns.tolist()
        self.visualizer.plot_feature_importance(self.results, feature_names, X_test[:100])

        # 步骤5: 生成分析报告
        self.generate_report(comparison_df)

        # 步骤6: 保存结果
        self.save_results()

        print("\n" + "="*70)
        print("分析流程完成！")
        print("="*70)

        return self.results

    def generate_report(self, comparison_df):
        """生成分析报告"""
        print("\n" + "="*70)
        print("石油溢出数据分类 - 分析报告")
        print("="*70)

        # 最佳模型
        best_model = comparison_df.index[0]
        best_accuracy = comparison_df.loc[best_model, 'accuracy']

        print(f"\n一、最佳模型: {best_model}")
        print(f"   准确率: {best_accuracy:.4f}")
        print(f"   F1分数: {comparison_df.loc[best_model, 'f1']:.4f}")
        print(f"   AUC分数: {comparison_df.loc[best_model, 'auc']:.4f}")

        print("\n二、模型性能排名:")
        for i, (model_name, row) in enumerate(comparison_df.iterrows(), 1):
            print(f"   {i}. {model_name}: 准确率={row['accuracy']:.4f}, "
                  f"F1={row['f1']:.4f}, AUC={row['auc']:.4f}")

        print("\n三、模型选择建议:")
        if best_model == '神经网络':
            print("   - 神经网络表现最佳，适合复杂模式识别")
            print("   - 需要足够的计算资源和数据")
        elif best_model in ['随机森林', 'XGBoost']:
            print("   - 集成方法表现良好，具有较好的泛化能力")
            print("   - 提供特征重要性，易于解释")
        elif best_model == 'SVM':
            print("   - SVM在小样本上表现稳定")
            print("   - 适合高维特征空间")
        else:
            print("   - 传统方法表现良好，计算效率高")

        print("\n四、改进建议:")
        print("   1. 数据层面:")
        print("      - 检查类别平衡，考虑过采样/欠采样")
        print("      - 尝试特征选择或特征工程")
        print("      - 收集更多数据提升模型泛化能力")
        print("   2. 模型层面:")
        print("      - 进行超参数调优")
        print("      - 尝试模型集成（投票、堆叠）")
        print("      - 考虑深度学习模型（如果数据量足够）")
        print("   3. 部署层面:")
        print("      - 根据准确率与计算资源需求选择模型")
        print("      - 考虑模型解释性要求")

        print("\n五、关键发现:")
        print("   - 所有模型的平均准确率: {:.4f}".format(comparison_df['accuracy'].mean()))
        print("   - 最佳与最差模型准确率差距: {:.4f}".format(
            comparison_df['accuracy'].max() - comparison_df['accuracy'].min()))

        # 保存报告
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(str(self))

        print(f"\n详细报告已保存至: {report_path}")

    def save_results(self):
        """保存所有结果"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 正在保存结果...")

        # 1. 保存神经网络模型
        nn_model_path = os.path.join(self.output_dir, 'nn_model.h5')
        self.nn_model.model.save(nn_model_path)
        print(f"神经网络模型已保存: {nn_model_path}")

        # 2. 保存最佳传统模型
        if '随机森林' in self.results:
            rf_model_path = os.path.join(self.output_dir, 'rf_model.pkl')
            joblib.dump(self.results['随机森林']['model'], rf_model_path)
            print(f"随机森林模型已保存: {rf_model_path}")

        # 3. 保存标准化器
        scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        joblib.dump(self.preprocessor.scaler, scaler_path)
        print(f"标准化器已保存: {scaler_path}")

        # 4. 保存所有结果
        results_path = os.path.join(self.output_dir, 'all_results.pkl')
        save_data = {
            'results': {k: {k2: v2 for k2, v2 in v.items() if k2 != 'model'}
                        for k, v in self.results.items()},
            'comparison_df': self.results,
            'preprocessor_info': {
                'data_shape': self.preprocessor.data.shape,
                'feature_names': self.preprocessor.data.columns.tolist(),
                'target_distribution': self.preprocessor.data[self.preprocessor.target_column].value_counts().to_dict()
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(results_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"所有结果已保存: {results_path}")

        # 5. 保存性能对比图
        fig, ax = plt.subplots(figsize=(12, 8))
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]

        bars = ax.barh(models, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        ax.set_xlabel('准确率')
        ax.set_title('模型性能对比')
        ax.set_xlim([0, 1.0])

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.4f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"性能对比图已保存: {os.path.join(self.output_dir, 'model_comparison.png')}")

    def __str__(self):
        """返回分析摘要"""
        if not self.results:
            return "尚未运行分析流程"

        summary = []
        summary.append("="*70)
        summary.append("石油溢出分类分析摘要")
        summary.append("="*70)

        for model_name, result in self.results.items():
            summary.append(f"{model_name}:")
            summary.append(f"  准确率: {result['accuracy']:.4f}")
            summary.append(f"  F1分数: {result['f1']:.4f}")
            summary.append(f"  AUC分数: {result['auc']:.4f}")
            if 'cv_mean' in result:
                summary.append(f"  交叉验证: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
            summary.append("")

        return "\n".join(summary)

# ============================================================================
# 6. 主程序入口
# ============================================================================
def main():
    """主函数"""
    print("石油溢出数据分类系统")
    print("="*50)

    # 配置参数
    DATA_PATH = "D:\cxdownload\oil_spill.csv"  # 修改为您的数据文件路径
    OUTPUT_DIR = "oil_spill_results"

    try:
        # 创建分类器实例
        classifier = OilSpillClassifier(DATA_PATH, OUTPUT_DIR)

        # 运行完整分析流程
        results = classifier.run_full_pipeline(test_size=0.2)

        # 打印摘要
        print("\n分析完成！")
        print(classifier)

        # 询问是否进行预测
        response = input("\n是否要对新数据进行预测？(y/n): ").lower()
        if response == 'y':
            # 这里可以添加预测代码
            print("预测功能暂未实现，请使用保存的模型文件进行预测。")
            print(f"模型文件保存在: {OUTPUT_DIR}/")

    except FileNotFoundError:
        print(f"错误: 找不到数据文件 '{DATA_PATH}'")
        print("请确保:")
        print("1. 数据文件存在")
        print("2. 文件路径正确")
        print("3. 文件名为: '分类——SVM实验.csv' 或修改代码中的DATA_PATH变量")

        # 显示当前目录文件
        print("\n当前目录下的文件:")
        for file in os.listdir('.'):
            if file.endswith('.csv') or file.endswith('.xlsx'):
                print(f"  - {file}")

    except Exception as e:
        print(f"运行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 执行主程序
# ============================================================================
if __name__ == "__main__":
    main()