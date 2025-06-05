import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

# Решение краевой задачи методом коллокаций с использованием B-сплайнов
def solve_boundary_value_problem_with_splines():
    a, b = 0, 1
    degree = 3
    num_inner_nodes = 30
    num_basis = num_inner_nodes + 2  # Учитываем граничные условия

    # Внутренние узлы (не включая границы)
    inner_knots = np.linspace(a, b, num_inner_nodes)[1:-1]
    # Полный вектор узлов с учётом повторения на границах
    knots = np.concatenate([[a] * (degree + 1), inner_knots, [b] * (degree + 1)])

    # Выбор точек коллокации
    collocation_points = [a]
    for i in range(len(inner_knots) - 1):
        collocation_points.append((inner_knots[i] + inner_knots[i + 1]) / 2)

    h = (b - a) / (num_inner_nodes + 1)
    for i in range(num_inner_nodes - len(inner_knots) + 1):
        collocation_points.append(a + (i + 1) * h)
    collocation_points.append(b)
    collocation_points = np.array(sorted(collocation_points[:num_basis]))

    A = np.zeros((num_basis, num_basis))  # Матрица системы
    F = np.zeros(num_basis)              # Вектор правых частей

    # Заполнение матрицы A и вектора F по внутренним точкам
    for j in range(1, num_basis - 1):
        x = collocation_points[j]
        for i in range(num_basis):
            coeffs = np.zeros(num_basis)
            coeffs[i] = 1
            spline = BSpline(knots, coeffs, degree)
            A[j, i] = spline.derivative(2)(x) - 3 * spline(x) / (x + 1)**2
        F[j] = -1.5 / np.sqrt(x + 1)

    # Граничное условие при x = 0: 3u(0) - u'(0) = 1
    for i in range(num_basis):
        coeffs = np.zeros(num_basis)
        coeffs[i] = 1
        spline = BSpline(knots, coeffs, degree)
        A[0, i] = 3 * spline(a) - spline.derivative(1)(a)
    F[0] = 1

    # Граничное условие при x = 1: u'(1) = sqrt(2)
    for i in range(num_basis):
        coeffs = np.zeros(num_basis)
        coeffs[i] = 1
        spline = BSpline(knots, coeffs, degree)
        A[num_basis - 1, i] = spline.derivative(1)(b)
    F[num_basis - 1] = np.sqrt(2)

    # Проверка обусловленности
    cond = np.linalg.cond(A)
    print(f"Число обусловленности матрицы системы: {cond:.16f}")
    if cond > 1e15:
        print("Добавление регуляризации из-за высокого числа обусловленности")
        A += np.eye(A.shape[0]) * 1e-10

    # Решение СЛАУ
    try:
        c = np.linalg.solve(A, F)
    except np.linalg.LinAlgError:
        print("Сингулярная матрица. Используется метод наименьших квадратов.")
        c, *_ = np.linalg.lstsq(A, F, rcond=None)

    solution_spline = BSpline(knots, c, degree)
    return solution_spline, knots, c, degree

# Точное аналитическое решение задачи
def exact_solution(x):
    return (2 / 3) * (x + 1) ** (3 / 2)

# Построение графика и вычисление ошибок
def plot_and_evaluate_results(spline, knots, coeffs, degree):
    x_vals = np.linspace(0, 1, 200)
    u_approx = spline(x_vals)
    u_exact = exact_solution(x_vals)

    error = np.abs(u_approx - u_exact)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error ** 2))

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, u_approx, 'r-', label='Приближённое решение')
    plt.plot(x_vals, u_exact, 'b--', label='Точное решение')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.title(f'Сравнение решений (СКО: {rms_error:.16f})')
    plt.grid(True)
    plt.tight_layout()

    return max_error, rms_error

# Главная функция
def main():
    spline, knots, coeffs, degree = solve_boundary_value_problem_with_splines()
    max_error, rms_error = plot_and_evaluate_results(spline, knots, coeffs, degree)

    print("\nКоэффициенты B-сплайна:")
    for i, c in enumerate(coeffs):
        print(f"c_{i} = {c:.16f}")

    print(f"\nМаксимальная погрешность: {max_error:.16f}")
    print(f"Среднеквадратичная погрешность: {rms_error:.16f}")

    u0, u1 = spline(0), spline(1)
    u0_deriv = spline.derivative(1)(0)
    u1_deriv = spline.derivative(1)(1)

    print("\nПроверка граничных условий:")
    print(f"3u(0) - u'(0) = {3 * u0 - u0_deriv:.16f} (должно быть 1)")
    print(f"u'(1) = {u1_deriv:.16f} (должно быть {np.sqrt(2):.16f})")

    print("\nСравнение в контрольных точках:")
    test_points = [0, 0.25, 0.5, 0.75, 1]
    print(f"{'x':>6} | {'u_approx':>16} | {'u_exact':>16} | {'|error|':>16}")
    print("-" * 66)
    for x in test_points:
        approx = spline(x)
        exact = exact_solution(x)
        error = abs(approx - exact)
        print(f"{x:6.2f} | {approx:16.12f} | {exact:16.12f} | {error:16.12f}")

    plt.show()

if __name__ == "__main__":
    main()
