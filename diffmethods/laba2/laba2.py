import numpy as np
import matplotlib.pyplot as plt

def solve_bvp(N, p, q, f, alpha0, beta0, gamma0, alpha1, beta1, gamma1):
    if N <= 1:
        raise ValueError("Число узлов N должно быть больше 1.")

    h = 1 / N
    x = np.linspace(0, 1, N + 1)

    a = np.zeros(N + 1)
    b = np.zeros(N + 1)
    c = np.zeros(N + 1)
    d = np.zeros(N + 1)

    for i in range(1, N):
        a[i] = 1 / h ** 2 - p(x[i]) / (2 * h)
        b[i] = -2 / h ** 2 + q(x[i])
        c[i] = 1 / h ** 2 + p(x[i]) / (2 * h)
        d[i] = f(x[i])

    if beta0 == 0:
        b[0] = alpha0
        d[0] = gamma0
    else:
        b[0] = alpha0 - beta0 / h
        c[0] = beta0 / h
        d[0] = gamma0

    if beta1 == 0:
        b[N] = alpha1
        d[N] = gamma1
    else:
        b[N] = alpha1 + beta1 / h
        a[N] = -beta1 / h
        d[N] = gamma1

    for i in range(1, N + 1):
        factor = a[i] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]

    y = np.zeros(N + 1)
    y[N] = d[N] / b[N]

    for i in range(N - 1, -1, -1):
        y[i] = (d[i] - c[i] * y[i + 1]) / b[i]

    return x, y

p = lambda x: -(x + 1)**2
q = lambda x: -2 / (x + 1)**2
f = lambda x: 1

N = 200
alpha0, beta0, gamma0 = 1, -1, 2
alpha1, beta1, gamma1 = 1, 0, 0.5

x, y = solve_bvp(N, p, q, f, alpha0, beta0, gamma0, alpha1, beta1, gamma1)

exact_solution = lambda x: 1 / (x + 1)
error = np.abs(y - exact_solution(x))

print("\nИсправленное решение:")
print("-" * 40)
print(f"{'x':<10} {'u(x)':<15} {'Ошибка':<15}")
print("-" * 40)
for xi, yi, ei in zip(x, y, error):
    print(f"{xi:<10.4f} {yi:<15.6f} {ei:<15.6f}")

plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', label='Численное решение')
plt.plot(x, exact_solution(x), 'r--', label='Приблизительное точное решение')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение численного и точного решения')
plt.legend()
plt.grid()
plt.show()