import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate  # Для красивого вывода таблицы в консоль

# Параметры задачи
l = 1.0   # Длина отрезка
T = 0.1   # Временной интервал
M = 20    # Число шагов по пространству
N = 6     # Число шагов по времени
a = 1.0   # Коэффициент теплопроводности

# Шаги сетки
h = l / M   # Шаг по пространству
tau = T / (20 * N)  # Уменьшенный шаг по времени для устойчивости (чтобы σ <= 0.5)
sigma = a**2 * tau / h**2

# Сетка по пространству и времени
x = np.linspace(0, l, M+1)
t = np.linspace(0, T, N+1)

# Начальные и граничные условия
def psi(x):
    return x**2 - x

def phi(x, t):
    return t**2 * x * ((1 - x)**2)

# Инициализация массивов для хранения решений
u_explicit = np.zeros((M+1, N+1))  # Явная схема
u_implicit = np.zeros((M+1, N+1))  # Неявная схема
u_crank = np.zeros((M+1, N+1))     # Схема Кранка-Николсона

# Начальные условия
u_explicit[:, 0] = psi(x)
u_implicit[:, 0] = psi(x)
u_crank[:, 0] = psi(x)

# Граничные условия
u_explicit[0, :] = u_explicit[-1, :] = 0
u_implicit[0, :] = u_implicit[-1, :] = 0
u_crank[0, :] = u_crank[-1, :] = 0

# Явная схема
for n in range(0, N):
    for i in range(1, M):
        u_explicit[i, n+1] = u_explicit[i, n] + sigma * (
            u_explicit[i+1, n] - 2 * u_explicit[i, n] + u_explicit[i-1, n]
        ) + tau * phi(x[i], t[n])

# Неявная схема (метод прогонки)
# Создание тридиагональной матрицы для решения системы уравнений
A = np.diag((1 + 2 * sigma) * np.ones(M-1)) + \
    np.diag(-sigma * np.ones(M-2), 1) + \
    np.diag(-sigma * np.ones(M-2), -1)

# Решение системы уравнений методом np.linalg.solve на каждом временном шаге
for n in range(0, N):
    b = u_implicit[1:M, n] + tau * phi(x[1:M], t[n])
    u_implicit[1:M, n+1] = np.linalg.solve(A, b)

# Схема Кранка-Николсона
# Матрицы для комбинированной схемы
A_cn = np.diag((1 + sigma) * np.ones(M-1)) + \
       np.diag(-sigma / 2 * np.ones(M-2), 1) + \
       np.diag(-sigma / 2 * np.ones(M-2), -1)

B_cn = np.diag((1 - sigma) * np.ones(M-1)) + \
       np.diag(sigma / 2 * np.ones(M-2), 1) + \
       np.diag(sigma / 2 * np.ones(M-2), -1)

# Решение системы линейных уравнений
for n in range(0, N):
    b = B_cn @ u_crank[1:M, n] + tau * phi(x[1:M], t[n])
    u_crank[1:M, n+1] = np.linalg.solve(A_cn, b)

# Построение графиков
plt.figure(figsize=(12, 8))
for n in range(N+1):
    if n in [0, 2, 4, 6]:  # Выводим графики для ключевых шагов
        plt.plot(x, u_explicit[:, n], label=f'Явная схема (шаг {n})')
        plt.plot(x, u_implicit[:, n], label=f'Неявная схема (шаг {n})')
        plt.plot(x, u_crank[:, n], label=f'Кранк-Николсон (шаг {n})')

plt.title('Сравнение методов решения')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()
plt.grid(True)
plt.show()

# Вывод таблицы в консоль
results = {
    'x': x,
    'Явная схема (t=0)': u_explicit[:, 0],
    'Явная схема (t=6)': u_explicit[:, -1],
    'Неявная схема (t=0)': u_implicit[:, 0],
    'Неявная схема (t=6)': u_implicit[:, -1],
    'Кранк-Николсон (t=0)': u_crank[:, 0],
    'Кранк-Николсон (t=6)': u_crank[:, -1]
}

df = pd.DataFrame(results)
print(tabulate(df, headers='keys', tablefmt='grid'))