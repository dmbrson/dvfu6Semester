import sympy as sp
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

x = sp.symbols('x')

u_exact = (2 / 3) * (x + 1)**(3 / 2)
u_exact_func = sp.lambdify(x, u_exact, 'numpy')

f_rhs = -3 / (2 * sp.sqrt(x + 1))

w = sp.Function('w')
w = (2 / 3) * (x + 1)**(3 / 2)

phi1 = x * (x - 1)**2
phi2 = x**2 * (x - 1)**2
phi3 = x**2 * (x - 1)**3

basis = [phi1, phi2, phi3]

basis_dbl_prime = [sp.diff(phi, x, 2) for phi in basis]

def L(u):
    return sp.diff(u, x, 2) - (3 / (x + 1)**2) * u

def compute_integral(expr):
    func = sp.lambdify(x, expr, 'numpy')
    result, _ = quad(func, 0, 1)
    return result

n = len(basis)
A = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        A[i, j] = compute_integral(L(basis[j]) * basis[i])

b = np.zeros(n)
for i in range(n):
    residual_rhs = f_rhs - L(w)
    b[i] = compute_integral(residual_rhs * basis[i])

C = np.linalg.solve(A, b)

u_approx = w + sum(C[i] * basis[i] for i in range(n))
u_approx_func = sp.lambdify(x, u_approx, 'numpy')

x_vals = np.linspace(0, 1, 300)
u_exact_vals = u_exact_func(x_vals)
u_approx_vals = u_approx_func(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, u_approx_vals, label='Приближенное решение', color='blue')
plt.plot(x_vals, u_exact_vals, label='Точное решение', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение приближенного и точного решений (с учётом граничных условий)')
plt.legend()
plt.grid(True)
plt.show()

for i, c in enumerate(C, 1):
    print(f'c{i} = {c:.6e}')

max_error = np.max(np.abs(u_exact_vals - u_approx_vals))
print(f'Максимальная ошибка: {max_error:.6f}')

u_approx_prime = sp.diff(u_approx, x)
u_approx_func_prime = sp.lambdify(x, u_approx_prime, 'numpy')

print('\nПроверка граничных условий:')
print(f'3u(0) - u\'(0) ≈ {3 * u_approx_func(0) - u_approx_func_prime(0):.6f} (должно быть ≈ 1)')
print(f'u\'(1) ≈ {u_approx_func_prime(1):.6f} (должно быть ≈ {np.sqrt(2):.6f})')
