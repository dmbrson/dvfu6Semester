import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM

np.random.seed(42)
n_samples = 1000
time = np.linspace(0, 10, n_samples)
data = np.sin(time)

plt.figure(figsize=(12, 4))
plt.plot(time, data, label='Исходная синусоида')
plt.title('Исходные данные')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.show()

window_size = 20
X = np.array([data[i:i+window_size] for i in range(n_samples - window_size)])

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X_binarized = (X_scaled > 0.5).astype(np.float32)

rbm = BernoulliRBM(n_components=50, learning_rate=0.05, batch_size=10, n_iter=100, random_state=42)
rbm.fit(X_binarized)

X_reconstructed_binarized = rbm.gibbs(X_binarized)

X_reconstructed_scaled = scaler.inverse_transform(X_reconstructed_binarized)

n_examples = 5
random_indices = np.random.choice(len(X), n_examples, replace=False)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(n_examples, 1, i+1)
    plt.plot(X[idx], 'b-', label='Исходные данные')
    plt.plot(X_reconstructed_scaled[idx], 'r--', label='Восстановленные данные')
    plt.title(f'Пример {i+1}')
    plt.legend()

plt.tight_layout()
plt.show()

mse = np.mean((X - X_reconstructed_scaled) ** 2)
print(f"Среднеквадратичная ошибка восстановления: {mse:.4f}")
