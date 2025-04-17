import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Параметры задачи
a = 1         # скорость волны
L = 1         # длина струны
T = 1         # общее время
M = 10        # количество отрезков по x
N = 20        # количество отрезков по t
h = L / M     # шаг по пространству
tau = T / N   # шаг по времени

# Сетки
x = np.linspace(0, L, M + 1)
t = np.linspace(0, T, N + 1)

# Решение u(x, t)
u = np.zeros((M + 1, N + 1))

print("a * tau / h =", a * tau / h, "< 1 =", a * tau / h < 1)

# Заданные функции
phi = lambda x: 0                 # начальная форма струны
psi = lambda x: 0                 # начальная скорость
g = lambda x, t: 0                # внешняя сила
gamma_0 = lambda t: -t            # левый конец
gamma_1 = lambda t: t**3          # правый конец

# Начальные условия
for m in range(M + 1):
    u[m, 0] = phi(x[m])

# Граничные условия
for n in range(N + 1):
    u[0, n] = gamma_0(t[n])
    u[M, n] = gamma_1(t[n])

# Второй временной слой
for m in range(1, M):
    u[m, 1] = (
        u[m, 0] + tau * psi(x[m]) +
        (tau ** 2 / 2) * (
            a ** 2 * (u[m - 1, 0] - 2 * u[m, 0] + u[m + 1, 0]) / h ** 2 +
            g(x[m], t[0])
        )
    )

# Остальные временные слои
for n in range(1, N):
    for m in range(1, M):
        u[m, n + 1] = (
            2 * u[m, n] - u[m, n - 1] +
            (a ** 2 * tau ** 2 / h ** 2) *
            (u[m - 1, n] - 2 * u[m, n] + u[m + 1, n]) +
            tau ** 2 * g(x[m], t[n])
        )

# Вывод максимум/минимум
print("Максимум u:", np.max(u))
print("Минимум u:", np.min(u))

# 3D-график
X, T_grid = np.meshgrid(t, x)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T_grid, u, cmap='viridis')
ax.set_title("Решение задачи (модифицированный вариант)")
ax.set_xlabel("Время t")
ax.set_ylabel("Координата x")
ax.set_zlabel("u(x, t)")
plt.tight_layout()
plt.show()

# Таблица значений
u_df = pd.DataFrame(
    u,
    index=[f"x = {x_val:.2f}" for x_val in x],
    columns=[f"t = {t_val:.2f}" for t_val in t]
)
print("Таблица значений u(x, t):")
print(u_df)

# Графики по срезам времени
time_indices = [0, 2, 4, 6, 8, 10]
cols = 3
rows = (len(time_indices) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
axes = axes.flatten()

for i, idx in enumerate(time_indices):
    axes[i].plot(x, u[:, idx], label=f"t = {t[idx]:.2f}")
    axes[i].set_title(f"Срез при t = {t[idx]:.2f}")
    axes[i].set_xlabel("x")
    axes[i].set_ylabel("u(x, t)")
    axes[i].legend()
    axes[i].grid(True)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
