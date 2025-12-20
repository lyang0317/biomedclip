import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# 1. 加载数据（假设文件名为 'cities.csv'）
data = pd.read_csv('D:\cxdownload\China_cities.csv', encoding='gbk')  # 文件格式：省,市,经度,纬度
coords = data[['东经', '北纬']].values

# 2. 定义聚类算法和参数
kmeans_params = {'n_clusters': [3, 4, 5, 6]}
spectral_params = {'n_clusters': [3, 4, 5, 6], 'gamma': [0.1, 1, 10]}

# 3. 聚类实验
results = []

# K-Means
for k in kmeans_params['n_clusters']:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(coords)
    silhouette = silhouette_score(coords, labels)
    results.append(('K-Means', k, None, silhouette))

# 谱聚类
for k in spectral_params['n_clusters']:
    for g in spectral_params['gamma']:
        spectral = SpectralClustering(n_clusters=k, gamma=g, random_state=42, affinity='rbf')
        labels = spectral.fit_predict(coords)
        silhouette = silhouette_score(coords, labels)
        results.append(('Spectral', k, g, silhouette))

# 4. 输出结果
print("聚类结果对比：")
print("算法\t簇数\tGamma\t轮廓系数")
for algo, k, g, sc in results:
    print(f"{algo}\t{k}\t{g}\t{sc:.4f}")

# 5. 选择最优模型（轮廓系数最高）
best = max(results, key=lambda x: x[3])
print(f"\n最优模型：{best[0]}, 簇数={best[1]}, gamma={best[2]}, 轮廓系数={best[3]:.4f}")

# 重新训练最优模型并可视化
if best[0] == 'K-Means':
    model = KMeans(n_clusters=best[1], random_state=42)
else:
    model = SpectralClustering(n_clusters=best[1], gamma=best[2], random_state=42, affinity='rbf')

labels = model.fit_predict(coords)

# 6. 绘图
plt.figure(figsize=(10, 8))
scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab20', s=30)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'City Clustering using {best[0]} (k={best[1]})')
plt.grid(True)
plt.show()