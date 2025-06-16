import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Загрузка данных
data = pd.read_csv('spotifydataset.csv')

# Выберем аудио-характеристики для кластеризации
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

X = data[audio_features].copy()
X.fillna(X.mean(), inplace=True)

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Определение оптимального числа кластеров
wcss = []
silhouette_scores = []
max_clusters = 20

for i in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Визуализация метода локтя
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters+1), wcss, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters+1), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Количество кластеров')
plt.ylabel('Score')
plt.tight_layout()
plt.show()

# Оптимальное количество кластеров
optimal_clusters = 7
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Добавляем метки кластеров в исходные данные
data['cluster'] = clusters

# Визуализация кластеров с помощью PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
data['pca1'] = principal_components[:, 0]
data['pca2'] = principal_components[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='viridis', s=60)
plt.title('Кластеризация треков Spotify (PCA)')
plt.show()

# Анализ характеристик кластеров
cluster_means = data.groupby('cluster')[audio_features].mean()

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, cmap='YlGnBu', annot=True, fmt='.2f')
plt.title('Средние значения аудио-характеристик по кластерам')
plt.show()

def recommend_songs(track_name, n_recommendations=5):
    try:
        cluster = data[data['track_name'] == track_name]['cluster'].values[0]
        same_cluster = data[(data['cluster'] == cluster) & (data['track_name'] != track_name)]
        recommendations = same_cluster.sort_values('track_popularity', ascending=False).head(n_recommendations)
        return recommendations[['track_name', 'artist_name', 'track_popularity']]
    except:
        return "Трек не найден в базе данных"

# Пример использования
print(recommend_songs("fukumean"))