import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Загрузка и подготовка данных с EDA
def load_and_prepare_data():
    df = pd.read_csv("csgo_games.csv")

    print("\nИнформация о датасете:")
    print(df.info())
    print(df.head())

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
    df[numeric_cols].hist(bins=20, figsize=(15, 10))
    plt.suptitle("Распределения признаков")
    plt.tight_layout()
    plt.show()

    corr = df[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                xticklabels=corr.columns, yticklabels=corr.columns)
    plt.title("Корреляционная матрица")
    plt.tight_layout()
    plt.show()

    df = pd.get_dummies(df)
    target = 'winner_t1'

    y = df[target].astype(int)
    df.drop(columns=[target], inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, df.columns

#RBF функции и слой
def gaussian_rbf(x, c, s):
    return torch.exp(-torch.sum((x - c) ** 2, dim=1) / (2 * s ** 2))

def multiquadric_rbf(x, c, s):
    return torch.sqrt(torch.sum((x - c) ** 2, dim=1) + s ** 2)

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, rbf_func='gaussian'):
        super().__init__()
        self.out_features = out_features
        self.centers = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.sigmas = nn.Parameter(torch.ones(out_features), requires_grad=False)
        self.rbf_func = gaussian_rbf if rbf_func == 'gaussian' else multiquadric_rbf

    def forward(self, x):
        return torch.stack([self.rbf_func(x, c, s) for c, s in zip(self.centers, self.sigmas)], dim=1)

class HybridRBFMLP(nn.Module):
    def __init__(self, in_features, rbf_units, mlp_hidden_sizes, out_features=1, rbf_func='gaussian'):
        super().__init__()
        self.rbf = RBFLayer(in_features, rbf_units, rbf_func)
        layers = []
        input_dim = rbf_units
        for hidden_size in mlp_hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        rbf_out = self.rbf(x)
        return self.mlp(rbf_out)

def init_centers_kmeans(X, n_centers):
    kmeans = KMeans(n_clusters=n_centers, n_init=10).fit(X)
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

def init_centers_random(X, n_centers):
    idx = np.random.choice(X.shape[0], n_centers, replace=False)
    return torch.tensor(X[idx], dtype=torch.float32)

def compute_sigma_global(centers):
    dists = euclidean_distances(centers, centers)
    return np.mean(dists)

def compute_sigma_nn(centers):
    dists = euclidean_distances(centers, centers)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    return np.mean(nearest)

#Обучение
def train_model(model, loader, optimizer, criterion, epochs=30):
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_prepare_data()

method = 'kmeans'
sigma_method = 'nn'

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.float32)

centers = init_centers_kmeans(X_train, 30) if method == 'kmeans' else init_centers_random(X_train, 30)
sigma = compute_sigma_nn(centers.numpy()) if sigma_method == 'nn' else compute_sigma_global(centers.numpy())

model = HybridRBFMLP(X_train.shape[1], rbf_units=30, mlp_hidden_sizes=[64, 32])
model.rbf.centers.data = centers
model.rbf.sigmas.data = torch.full((30,), sigma)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

train_model(model, train_loader, optimizer, criterion, epochs=30)

model.eval()
with torch.no_grad():
    val_preds = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32))).squeeze().numpy()
    val_bin = (val_preds > 0.5).astype(int)
    acc = accuracy_score(y_val, val_bin)
    f1 = f1_score(y_val, val_bin)
    prec = precision_score(y_val, val_bin)
    recall = recall_score(y_val, val_bin)

print("\nОценка модели:")
print(f"точность: {acc:.4f}")
print(f"f1-мера: {f1:.4f}")

#Сравнение RBF функций
for rbf_name in ['gaussian', 'multiquadric']:
    model = HybridRBFMLP(X_train.shape[1], 30, [64, 32], rbf_func=rbf_name)
    model.rbf.centers.data = centers
    model.rbf.sigmas.data = torch.full((30,), sigma)
    train_model(model, train_loader, optimizer, criterion, epochs=10)
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32))).squeeze().numpy()
        bin_preds = (preds > 0.5).astype(int)
        acc_rb = accuracy_score(y_val, bin_preds)
        f1_rb = f1_score(y_val, bin_preds)
    print(f"\nФункция: {rbf_name}")
    print(f"точность: {acc_rb:.4f}")
    print(f"f1-мера: {f1_rb:.4f}")

# Проклятие размерности
dimensionality_accuracies = []
dimensionality_times = []
dimensions = [5, 10, 20, 50, 100]

print("\nИсследование проклятия размерности:")
for k in dimensions:
    X_k = X_train[:, :k]
    Xv_k = X_val[:, :k]
    centers_k = init_centers_kmeans(X_k, 10)
    sigma_k = compute_sigma_nn(centers_k.numpy())
    model_k = HybridRBFMLP(k, 10, [32])
    model_k.rbf.centers.data = centers_k
    model_k.rbf.sigmas.data = torch.full((10,), sigma_k)

    loader_k = DataLoader(
        TensorDataset(torch.tensor(X_k, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)),
        batch_size=32, shuffle=True)

    start_time = time.time()
    train_model(model_k, loader_k, optimizer, criterion, epochs=10)
    elapsed_time = time.time() - start_time

    with torch.no_grad():
        preds_k = torch.sigmoid(model_k(torch.tensor(Xv_k, dtype=torch.float32))).squeeze().numpy()
        bin_k = (preds_k > 0.5).astype(int)
        accuracy = accuracy_score(y_val, bin_k)
        dimensionality_accuracies.append(accuracy)
        dimensionality_times.append(elapsed_time)
        print(f"Количество признаков: {k}")
        print(f"Точность: {accuracy:.4f}, Время обучения: {elapsed_time:.2f} сек")

# размерность vs точность
plt.figure(figsize=(8, 5))
plt.plot(dimensions, dimensionality_accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Количество признаков')
plt.ylabel('Точность')
plt.title('Влияние размерности на точность модели')
plt.grid(True)
plt.show()

# размерность vs время
plt.figure(figsize=(8, 5))
plt.plot(dimensions, dimensionality_times, marker='o', linestyle='-', color='orange')
plt.xlabel('Количество признаков')
plt.ylabel('Время обучения (сек)')
plt.title('Влияние размерности на время обучения')
plt.grid(True)
plt.show()

# зависимость от числа центров
center_counts = [10, 20, 50, 100]
center_accuracies = []
center_times = []

X_fixed = X_train[:, :20]
Xv_fixed = X_val[:, :20]

print("\nЗависимость от числа центров:")
for c in center_counts:
    centers_c = init_centers_kmeans(X_fixed, c)
    sigma_c = compute_sigma_nn(centers_c.numpy())
    model_c = HybridRBFMLP(20, c, [32])
    model_c.rbf.centers.data = centers_c
    model_c.rbf.sigmas.data = torch.full((c,), sigma_c)

    loader_c = DataLoader(
        TensorDataset(torch.tensor(X_fixed, dtype=torch.float32),
                      torch.tensor(y_train.values, dtype=torch.float32)),
        batch_size=32, shuffle=True)

    start_time = time.time()
    train_model(model_c, loader_c, optimizer, criterion, epochs=10)
    elapsed_time = time.time() - start_time

    with torch.no_grad():
        preds_c = torch.sigmoid(model_c(torch.tensor(Xv_fixed, dtype=torch.float32))).squeeze().numpy()
        bin_c = (preds_c > 0.5).astype(int)
        accuracy = accuracy_score(y_val, bin_c)
        center_accuracies.append(accuracy)
        center_times.append(elapsed_time)
        print(f"Количество центров: {c}")
        print(f"Точность: {accuracy:.4f}, Время обучения: {elapsed_time:.2f} сек")

# центры vs точность
plt.figure(figsize=(8, 5))
plt.plot(center_counts, center_accuracies, marker='o', linestyle='-', color='g')
plt.xlabel('Количество центров')
plt.ylabel('Точность')
plt.title('Влияние количества центров на точность модели')
plt.grid(True)
plt.show()

# центры vs время
plt.figure(figsize=(8, 5))
plt.plot(center_counts, center_times, marker='o', linestyle='-', color='red')
plt.xlabel('Количество центров')
plt.ylabel('Время обучения (сек)')
plt.title('Влияние количества центров на время обучения')
plt.grid(True)
plt.show()


#Сравнение с другими моделями
print("Сравнение с другими моделями:")

# Логистическая регрессия
clf_lr = LogisticRegression(max_iter=100)
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_val)
print("\nЛогистическая регрессия")
print(f"точность: {accuracy_score(y_val, pred_lr):.4f}")
print(f"f1-мера: {f1_score(y_val, pred_lr):.4f}")

# Лес
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
pred_rf = clf_rf.predict(X_val)
print("\nСлучайный лес")
print(f"точность: {accuracy_score(y_val, pred_rf):.4f}")
print(f"f1-мера: {f1_score(y_val, pred_rf):.4f}")

# SVM
clf_svm = SVC()
clf_svm.fit(X_train, y_train)
pred_svm = clf_svm.predict(X_val)
print("\nSVM")
print(f"точность: {accuracy_score(y_val, pred_svm):.4f}")
print(f"f1-мера: {f1_score(y_val, pred_svm):.4f}")

# XGBoost
clf_xgb = XGBClassifier(eval_metric='logloss')
clf_xgb.fit(X_train, y_train)
pred_xgb = clf_xgb.predict(X_val)
print("\nГрадиентный бустинг (XGBoost)")
print(f"точность: {accuracy_score(y_val, pred_xgb):.4f}")
print(f"f1-мера: {f1_score(y_val, pred_xgb):.4f}")