import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Заголовок
print("Решение дифференциального уравнения методом коллокации:")
print("u'' - 3u / (x+1)^2 = -1.5 / (x+1)^(1/2)")
print("с граничными условиями: 3u(0) - u'(0) = 1, u'(1) = sqrt(2)")

# Переменные
x = sp.Symbol('x')
c0, c1, c2, c3 = sp.symbols('c0 c1 c2 c3')

# Полином
u = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3

# Производные
u_prime = sp.diff(u, x)
u_double_prime = sp.diff(u_prime, x)

# Правая часть уравнения
f = -1.5 / sp.sqrt(x + 1)

# Остаток (остаточная функция)
R = u_double_prime - (3 * u) / (x + 1)**2 - f

# Граничные условия
eq1 = 3 * u.subs(x, 0) - u_prime.subs(x, 0) - 1        # 3u(0) - u'(0) = 1
eq2 = u_prime.subs(x, 1) - np.sqrt(2)                  # u'(1) = sqrt(2)

# Коллокационные точки
collocation_points = [0.3, 0.7]
eq3 = R.subs(x, collocation_points[0])
eq4 = R.subs(x, collocation_points[1])

# Система уравнений
equations = [eq1, eq2, eq3, eq4]
variables = [c0, c1, c2, c3]

# Составляем матрицу A и вектор b
A = np.zeros((4, 4))
b = np.zeros(4)

for i, eq in enumerate(equations):
    for j, var in enumerate(variables):
        coeff = eq.coeff(var) if var in eq.free_symbols else 0
        A[i, j] = float(coeff)
    const = eq.subs({v: 0 for v in variables})
    b[i] = -float(const)

# Решение системы
coeffs = np.linalg.solve(A, b)
c0_val, c1_val, c2_val, c3_val = coeffs

# Вывод коэффициентов
print("\nКоэффициенты приближенного решения:")
print(f"c0 = {c0_val:.6f}")
print(f"c1 = {c1_val:.6f}")
print(f"c2 = {c2_val:.6f}")
print(f"c3 = {c3_val:.6f}")

# Функция приближённого решения
def u_approx(x_val):
    return c0_val + c1_val * x_val + c2_val * x_val ** 2 + c3_val * x_val ** 3

# Точное решение
def u_exact(x_val):
    return (2 / 3) * (x_val + 1) ** (3 / 2)

print("\nТочное решение: u(x) = (2/3)*(x + 1)^(3/2)")

# Сравнение в точках
print("\nСравнение точного и приближенного решений:")
points = [0, 0.25, 0.5, 0.75, 1]
for p in points:
    approx = u_approx(p)
    exact = u_exact(p)
    print(f"x = {p:.2f}: приближ. = {approx:.6f}, точн. = {exact:.6f}, ошибка = {approx - exact:.6f}")

# Построение графиков
x_vals = np.linspace(0, 1, 200)
u_approx_vals = u_approx(x_vals)
u_exact_vals = u_exact(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, u_approx_vals, label='Приближенное решение', color='blue')
plt.plot(x_vals, u_exact_vals, label='Точное решение', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Сравнение приближенного и точного решений')
plt.legend()
plt.grid(True)
plt.show()

# Максимальная ошибка
error = np.abs(u_exact_vals - u_approx_vals)
print(f"\nМаксимальная ошибка: {np.max(error):.6f}")
