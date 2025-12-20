import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据准备和预处理
class GeologicalImageLoader:
    """地质图像数据加载器"""

    def __init__(self, data_dir):
        """
        初始化
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.image_paths = []
        self.image_names = []
        self.labels = []

    def load_images(self, max_per_class=2, img_size=(64, 64)):
        """
        加载图像数据
        Args:
            max_per_class: 每个类别最多加载的图像数
            img_size: 图像缩放尺寸
        Returns:
            images: 图像数据数组
            labels: 真实标签
            image_names: 图像名称列表
        """
        print("开始加载地质图像数据...")

        images = []
        categories = {}

        # 遍历目录
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    file_path = os.path.join(root, file)

                    # 提取类别信息
                    category = os.path.basename(root)

                    # 处理"+"和"-"符号
                    if '_+' in file or '_-' in file:
                        base_name = file.split('_+')[0].split('_-')[0]
                    else:
                        base_name = file.split('.')[0]

                    # 限制每个类别的图像数量
                    if category not in categories:
                        categories[category] = 0

                    if categories[category] >= max_per_class:
                        continue

                    # 读取并预处理图像
                    try:
                        img = cv2.imread(file_path)
                        if img is None:
                            continue

                        # 转换为RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # 调整大小
                        img_resized = cv2.resize(img_rgb, img_size)

                        # 展平为特征向量
                        img_flattened = img_resized.flatten()

                        images.append(img_flattened)
                        self.image_paths.append(file_path)
                        self.image_names.append(file)

                        # 使用目录名作为标签
                        self.labels.append(category)

                        categories[category] += 1

                    except Exception as e:
                        print(f"处理图像 {file_path} 时出错: {e}")
                        continue

        print(f"成功加载 {len(images)} 张图像")
        print(f"类别分布: {categories}")

        return np.array(images), np.array(self.labels), self.image_names

    def extract_features(self, images, method='color_histogram'):
        """
        提取图像特征
        Args:
            images: 原始图像数据
            method: 特征提取方法
        Returns:
            features: 提取的特征
        """
        print(f"使用 {method} 方法提取特征...")

        if method == 'color_histogram':
            # 颜色直方图特征
            features = []
            for img in images:
                # 假设images是展平后的，需要重新reshape
                if len(img) == 64*64*3:  # RGB图像
                    img_reshaped = img.reshape(64, 64, 3)

                    # 计算每个通道的直方图
                    hist_r = cv2.calcHist([img_reshaped[:,:,0]], [0], None, [16], [0, 256]).flatten()
                    hist_g = cv2.calcHist([img_reshaped[:,:,1]], [0], None, [16], [0, 256]).flatten()
                    hist_b = cv2.calcHist([img_reshaped[:,:,2]], [0], None, [16], [0, 256]).flatten()

                    # 合并特征
                    hist_features = np.concatenate([hist_r, hist_g, hist_b])
                    features.append(hist_features)

            return np.array(features)

        elif method == 'raw_pixels':
            # 直接使用像素值（降维后）
            from sklearn.decomposition import PCA
            pca = PCA(n_components=100, random_state=42)
            features = pca.fit_transform(images)
            return features

        else:
            # 默认返回原始数据（可能需要降维）
            return images

# 2. 聚类实验主函数
def geological_image_clustering_experiment(data_dir, n_clusters_range=None, gamma_range=None):
    """
    地质图像聚类实验
    """
    # 初始化
    if n_clusters_range is None:
        n_clusters_range = [2, 3, 4, 5, 6]
    if gamma_range is None:
        gamma_range = [0.1, 1, 10]

    # 加载数据
    loader = GeologicalImageLoader(data_dir)
    images, true_labels, image_names = loader.load_images(max_per_class=2, img_size=(64, 64))

    # 提取特征
    features = loader.extract_features(images, method='color_histogram')

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 存储结果
    results = []

    print("\n" + "="*50)
    print("开始聚类实验")
    print("="*50)

    # 实验1: K-Means聚类
    print("\n=== K-Means聚类实验 ===")
    for n_clusters in n_clusters_range:
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(features_scaled)

            # 计算评价指标
            silhouette = silhouette_score(features_scaled, pred_labels)

            # 计算ARI（如果有真实标签）
            ari = None
            if true_labels is not None and len(np.unique(true_labels)) > 1:
                # 将字符串标签转换为数值
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                true_labels_numeric = le.fit_transform(true_labels)
                ari = adjusted_rand_score(true_labels_numeric, pred_labels)

            results.append({
                '算法': 'K-Means',
                '簇数': n_clusters,
                '参数': 'None',
                '轮廓系数': silhouette,
                'ARI': ari
            })

            print(f"K={n_clusters}: 轮廓系数={silhouette:.4f}, ARI={ari if ari is not None else 'N/A'}")

        except Exception as e:
            print(f"K={n_clusters} 时出错: {e}")

    # 实验2: 谱聚类
    print("\n=== 谱聚类实验 ===")
    for n_clusters in n_clusters_range:
        for gamma in gamma_range:
            try:
                spectral = SpectralClustering(
                    n_clusters=n_clusters,
                    gamma=gamma,
                    affinity='rbf',
                    random_state=42
                )
                pred_labels = spectral.fit_predict(features_scaled)

                # 计算评价指标
                silhouette = silhouette_score(features_scaled, pred_labels)

                # 计算ARI
                ari = None
                if true_labels is not None and len(np.unique(true_labels)) > 1:
                    le = LabelEncoder()
                    true_labels_numeric = le.fit_transform(true_labels)
                    ari = adjusted_rand_score(true_labels_numeric, pred_labels)

                results.append({
                    '算法': '谱聚类',
                    '簇数': n_clusters,
                    '参数': f'gamma={gamma}',
                    '轮廓系数': silhouette,
                    'ARI': ari
                })

                print(f"K={n_clusters}, gamma={gamma}: 轮廓系数={silhouette:.4f}, ARI={ari if ari is not None else 'N/A'}")

            except Exception as e:
                print(f"K={n_clusters}, gamma={gamma} 时出错: {e}")

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 找到最优模型
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        # 使用ARI选择最优
        best_idx = results_df['ARI'].idxmax()
        print("\n基于ARI选择最优模型")
    else:
        # 使用轮廓系数选择最优
        best_idx = results_df['轮廓系数'].idxmax()
        print("\n基于轮廓系数选择最优模型")

    best_model = results_df.iloc[best_idx]

    print("\n" + "="*50)
    print("最佳聚类结果")
    print("="*50)
    print(f"算法: {best_model['算法']}")
    print(f"簇数: {best_model['簇数']}")
    print(f"参数: {best_model['参数']}")
    print(f"轮廓系数: {best_model['轮廓系数']:.4f}")
    if best_model['ARI'] is not None:
        print(f"ARI: {best_model['ARI']:.4f}")

    return results_df, best_model, features_scaled, true_labels, image_names

# 3. 可视化函数
def visualize_clustering_results(results_df, best_model, features, true_labels, pred_labels, image_names):
    """
    可视化聚类结果
    """
    print("\n生成可视化结果...")

    # 1. 指标对比图
    fig = plt.figure(figsize=(15, 10))

    # 子图1: 不同算法的轮廓系数对比
    ax1 = plt.subplot(2, 3, 1)
    kmeans_results = results_df[results_df['算法'] == 'K-Means']
    spectral_results = results_df[results_df['算法'] == '谱聚类']

    ax1.plot(kmeans_results['簇数'], kmeans_results['轮廓系数'], 'bo-', label='K-Means', linewidth=2)
    for n_clusters in np.unique(spectral_results['簇数']):
        cluster_data = spectral_results[spectral_results['簇数'] == n_clusters]
        ax1.plot(cluster_data['簇数'], cluster_data['轮廓系数'], 'ro-', label=f'谱聚类(K={n_clusters})',
                 linewidth=1, alpha=0.5)

    ax1.set_xlabel('簇数(K)')
    ax1.set_ylabel('轮廓系数')
    ax1.set_title('不同算法的轮廓系数对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2: ARI对比（如果有）
    ax2 = plt.subplot(2, 3, 2)
    if results_df['ARI'].notna().any():
        kmeans_ari = kmeans_results['ARI'].dropna()
        spectral_ari = spectral_results['ARI'].dropna()

        if len(kmeans_ari) > 0:
            ax2.plot(kmeans_results['簇数'][:len(kmeans_ari)], kmeans_ari, 'bo-',
                     label='K-Means', linewidth=2)

        if len(spectral_ari) > 0:
            # 简化显示
            ax2.plot([], [], 'ro-', label='谱聚类', linewidth=2)
            for idx, row in spectral_results.iterrows():
                if not pd.isna(row['ARI']):
                    ax2.plot(row['簇数'], row['ARI'], 'ro', alpha=0.7)

        ax2.set_xlabel('簇数(K)')
        ax2.set_ylabel('调整兰德系数(ARI)')
        ax2.set_title('不同算法的ARI对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '无真实标签\n无法计算ARI',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=12)
        ax2.set_title('ARI对比')

    # 子图3: 参数影响（谱聚类的gamma参数）
    ax3 = plt.subplot(2, 3, 3)
    spectral_results = results_df[results_df['算法'] == '谱聚类']

    for gamma in [0.1, 1, 10]:
        gamma_data = spectral_results[spectral_results['参数'] == f'gamma={gamma}']
        if len(gamma_data) > 0:
            ax3.plot(gamma_data['簇数'], gamma_data['轮廓系数'], 'o-',
                     label=f'gamma={gamma}', linewidth=2)

    ax3.set_xlabel('簇数(K)')
    ax3.set_ylabel('轮廓系数')
    ax3.set_title('Gamma参数对谱聚类的影响')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 子图4: PCA降维可视化聚类结果
    ax4 = plt.subplot(2, 3, 4)
    from sklearn.decomposition import PCA

    # 使用PCA降维到2D
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)

    scatter = ax4.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=pred_labels, cmap='tab20', s=50, alpha=0.7)
    ax4.set_xlabel('PCA主成分1')
    ax4.set_ylabel('PCA主成分2')
    ax4.set_title(f'聚类结果可视化 (K={len(np.unique(pred_labels))})')
    plt.colorbar(scatter, ax=ax4, label='聚类标签')

    # 子图5: 真实标签与预测标签对比（如果有真实标签）
    ax5 = plt.subplot(2, 3, 5)
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        true_labels_numeric = le.fit_transform(true_labels)

        scatter = ax5.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=true_labels_numeric, cmap='tab20', s=50, alpha=0.7)
        ax5.set_xlabel('PCA主成分1')
        ax5.set_ylabel('PCA主成分2')
        ax5.set_title('真实类别分布')
        plt.colorbar(scatter, ax=ax5, label='真实标签')
    else:
        ax5.text(0.5, 0.5, '无真实标签数据',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax5.transAxes, fontsize=12)
        ax5.set_title('真实类别分布')

    # 子图6: 最佳模型信息
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    info_text = f"最佳模型信息:\n\n"
    info_text += f"算法: {best_model['算法']}\n"
    info_text += f"簇数: {best_model['簇数']}\n"
    info_text += f"参数: {best_model['参数']}\n"
    info_text += f"轮廓系数: {best_model['轮廓系数']:.4f}\n"
    if best_model['ARI'] is not None:
        info_text += f"ARI: {best_model['ARI']:.4f}"

    ax6.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center')

    plt.suptitle('地质图像聚类实验结果分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 2. 显示示例图像
    display_sample_images(pred_labels, image_names, best_model['簇数'])

def display_sample_images(pred_labels, image_names, n_clusters, max_samples=5):
    """
    显示每个簇的示例图像
    """
    print(f"\n显示每个簇的示例图像（最多{max_samples}张）...")

    unique_labels = np.unique(pred_labels)
    n_rows = len(unique_labels)

    fig, axes = plt.subplots(n_rows, max_samples, figsize=(15, n_rows*2))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, cluster_id in enumerate(unique_labels):
        # 获取属于当前簇的图像索引
        cluster_indices = np.where(pred_labels == cluster_id)[0]

        # 随机选择最多max_samples个样本
        if len(cluster_indices) > max_samples:
            sample_indices = np.random.choice(cluster_indices, max_samples, replace=False)
        else:
            sample_indices = cluster_indices

        for j, idx in enumerate(sample_indices):
            if j >= max_samples:
                break

            try:
                # 读取图像
                img_path = image_names[idx]  # 注意：这里需要实际的图像路径
                # 在实际应用中，这里应该加载图像
                # img = cv2.imread(full_path)
                # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 这里用占位符代替
                placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
                placeholder[:, :, 0] = np.random.randint(0, 255)  # R
                placeholder[:, :, 1] = np.random.randint(0, 255)  # G
                placeholder[:, :, 2] = np.random.randint(0, 255)  # B

                axes[i, j].imshow(placeholder)
                axes[i, j].axis('off')

                if j == 0:
                    axes[i, j].set_title(f'簇{cluster_id}\n({len(cluster_indices)}张)', fontsize=10)
            except:
                pass

    plt.suptitle('各聚类簇的示例图像', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# 4. 如果没有实际数据，使用模拟数据进行测试
def create_sample_geological_data(n_samples=100, n_features=768):
    """
    创建模拟的地质图像数据
    """
    print("创建模拟地质图像数据...")

    np.random.seed(42)

    # 创建4个不同的"矿物类别"
    n_clusters = 4
    cluster_centers = []

    # 不同类别的特征中心
    for i in range(n_clusters):
        center = np.random.normal(i*50, 20, n_features)
        cluster_centers.append(center)

    # 生成数据
    X = []
    true_labels = []
    image_names = []

    for i in range(n_samples):
        # 随机选择一个类别
        cluster_id = i % n_clusters
        center = cluster_centers[cluster_id]

        # 生成数据点
        point = center + np.random.normal(0, 10, n_features)
        X.append(point)
        true_labels.append(cluster_id)

        # 生成模拟图像名称
        mineral_types = ['石英', '长石', '云母', '方解石']
        light_types = ['+', '-']
        mineral = mineral_types[cluster_id]
        light = np.random.choice(light_types)
        image_names.append(f"{mineral}_sample_{i}_{light}.jpg")

    X = np.array(X)
    true_labels = np.array(true_labels)

    print(f"创建了 {n_samples} 个样本，{n_clusters} 个类别")
    print(f"类别分布: {np.bincount(true_labels)}")

    return X, true_labels, image_names

# 5. 主程序
if __name__ == "__main__":
    # 设置数据路径
    data_dir = "D:\cxdownload\stone"  # 修改为实际路径

    if not os.path.exists(data_dir):
        print(f"警告: 数据目录 '{data_dir}' 不存在")
        print("使用模拟数据进行演示...")

        # 使用模拟数据
        features_scaled, true_labels, image_names = create_sample_geological_data(
            n_samples=120, n_features=768
        )

        # 手动创建结果数据
        results = []
        n_clusters_range = [2, 3, 4, 5, 6]
        gamma_range = [0.1, 1, 10]

        # 模拟K-Means结果
        for k in n_clusters_range:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
            pred_labels = kmeans.fit_predict(features_scaled)

            silhouette = silhouette_score(features_scaled, pred_labels)
            ari = adjusted_rand_score(true_labels, pred_labels)

            results.append({
                '算法': 'K-Means',
                '簇数': k,
                '参数': 'None',
                '轮廓系数': silhouette,
                'ARI': ari
            })

        # 模拟谱聚类结果
        for k in n_clusters_range:
            for gamma in gamma_range:
                from sklearn.cluster import SpectralClustering
                spectral = SpectralClustering(
                    n_clusters=k,
                    gamma=gamma,
                    affinity='rbf',
                    random_state=42
                )
                pred_labels = spectral.fit_predict(features_scaled)

                silhouette = silhouette_score(features_scaled, pred_labels)
                ari = adjusted_rand_score(true_labels, pred_labels)

                results.append({
                    '算法': '谱聚类',
                    '簇数': k,
                    '参数': f'gamma={gamma}',
                    '轮廓系数': silhouette,
                    'ARI': ari
                })

        results_df = pd.DataFrame(results)
        best_idx = results_df['轮廓系数'].idxmax()
        best_model = results_df.iloc[best_idx]

        # 重新计算最佳模型的聚类结果
        if best_model['算法'] == 'K-Means':
            final_model = KMeans(n_clusters=best_model['簇数'], random_state=42)
        else:
            gamma = float(best_model['参数'].split('=')[1])
            final_model = SpectralClustering(
                n_clusters=best_model['簇数'],
                gamma=gamma,
                affinity='rbf',
                random_state=42
            )

        final_labels = final_model.fit_predict(features_scaled)

        # 可视化
        visualize_clustering_results(results_df, best_model, features_scaled,
                                     true_labels, final_labels, image_names)

    else:
        # 使用真实数据
        results_df, best_model, features_scaled, true_labels, image_names = \
            geological_image_clustering_experiment(data_dir)

        # 重新训练最佳模型
        if best_model['算法'] == 'K-Means':
            final_model = KMeans(n_clusters=best_model['簇数'], random_state=42)
        else:
            gamma = float(best_model['参数'].split('=')[1])
            final_model = SpectralClustering(
                n_clusters=best_model['簇数'],
                gamma=gamma,
                affinity='rbf',
                random_state=42
            )

        final_labels = final_model.fit_predict(features_scaled)

        # 可视化
        visualize_clustering_results(results_df, best_model, features_scaled,
                                     true_labels, final_labels, image_names)

    # 保存结果
    print("\n保存实验结果...")
    results_df.to_csv('地质图像聚类实验结果.csv', index=False, encoding='utf-8-sig')
    print("结果已保存到 '地质图像聚类实验结果.csv'")