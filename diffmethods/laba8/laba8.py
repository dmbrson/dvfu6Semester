import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12


def exact_solution(x, y, t, solution_type=3):
    return t + x ** 2 + y ** 2


def source_term(x, y, t, solution_type=3):
    return 1 - 4  # ∂u/∂t = Δu + f => 1 = (2 + 2) + f => f = -3


def fractional_step_method(solution_type=3, nx=10, ny=10, T=1.0, nt=100):
    lx, ly = 1.0, 1.0
    hx = lx / nx
    hy = ly / ny
    x = np.linspace(0, lx, nx + 1)
    y = np.linspace(0, ly, ny + 1)
    dt = T / nt

    Lambda1 = 1 / hx ** 2
    Lambda2 = 1 / hy ** 2

    error_list = []

    # Начальное условие
    v = np.zeros((ny + 1, nx + 1))
    for i in range(ny + 1):
        for j in range(nx + 1):
            v[i, j] = exact_solution(x[j], y[i], 0, solution_type)

    for n in range(nt):
        t_n = n * dt
        t_np1 = (n + 1) * dt

        # Первый дробный шаг: n → n+1/2 по x
        v_half = np.zeros_like(v)
        for i in range(1, ny):
            a = np.zeros(nx + 1)
            b = np.ones(nx + 1)
            c = np.zeros(nx + 1)
            d = np.zeros(nx + 1)

            for j in range(1, nx):
                phi = source_term(x[j], y[i], t_n, solution_type)
                a[j] = -dt * Lambda1 / 2
                b[j] = 1 + dt * Lambda1
                c[j] = -dt * Lambda1 / 2

                d[j] = v[i, j] + dt / 2 * (
                        Lambda1 * (v[i, j + 1] - 2 * v[i, j] + v[i, j - 1]) +
                        Lambda2 * (v[i + 1, j] - 2 * v[i, j] + v[i - 1, j]) + phi
                )

            # Граничные условия по x
            b[0], d[0] = 1, exact_solution(0, y[i], t_np1, solution_type)
            b[nx], d[nx] = 1, exact_solution(lx, y[i], t_np1, solution_type)

            v_half[i, :] = tridiagonal_solver(a, b, c, d, nx + 1)

        # Граничные условия по y для промежуточного решения
        for j in range(nx + 1):
            v_half[0, j] = exact_solution(x[j], 0, t_np1, solution_type)
            v_half[ny, j] = exact_solution(x[j], ly, t_np1, solution_type)

        # Второй дробный шаг: n+1/2 → n+1 по y
        v_new = np.zeros_like(v)
        for j in range(1, nx):
            a = np.zeros(ny + 1)
            b = np.ones(ny + 1)
            c = np.zeros(ny + 1)
            d = np.zeros(ny + 1)

            for i in range(1, ny):
                a[i] = -dt * Lambda2 / 2
                b[i] = 1 + dt * Lambda2
                c[i] = -dt * Lambda2 / 2

                d[i] = v_half[i, j] + dt / 2 * (
                        Lambda1 * (v_half[i, j + 1] - 2 * v_half[i, j] + v_half[i, j - 1]) +
                        Lambda2 * (v_half[i + 1, j] - 2 * v_half[i, j] + v_half[i - 1, j]) +
                        source_term(x[j], y[i], t_np1, solution_type)
                )

            # Граничные условия по y
            b[0], d[0] = 1, exact_solution(x[j], 0, t_np1, solution_type)
            b[ny], d[ny] = 1, exact_solution(x[j], ly, t_np1, solution_type)

            v_new[:, j] = tridiagonal_solver(a, b, c, d, ny + 1)

        # Граничные условия по x для окончательного решения
        for i in range(ny + 1):
            v_new[i, 0] = exact_solution(0, y[i], t_np1, solution_type)
            v_new[i, nx] = exact_solution(lx, y[i], t_np1, solution_type)

        v = v_new.copy()

        # Вычисление точного решения и ошибки на текущем шаге
        t_np1 = (n + 1) * dt
        u_ex = np.zeros_like(v)
        for i in range(ny + 1):
            for j in range(nx + 1):
                u_ex[i, j] = exact_solution(x[j], y[i], t_np1, solution_type)

        l2_err, _ = calculate_error(v, u_ex)
        error_list.append((t_np1, l2_err))  # сохраняем время и ошибку

    # Точное решение в конечный момент времени
    u_exact = np.zeros_like(v)
    for i in range(ny + 1):
        for j in range(nx + 1):
            u_exact[i, j] = exact_solution(x[j], y[i], T, solution_type)

    return x, y, v, u_exact, error_list


def tridiagonal_solver(a, b, c, d, n):
    # Прямой ход
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]

    # Обратный ход
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


def calculate_error(u_numerical, u_exact):
    """Вычисление ошибок"""
    diff = u_numerical - u_exact
    l2_error = np.sqrt(np.mean(diff ** 2))
    max_error = np.max(np.abs(diff))
    return l2_error, max_error


def plot_solution(x, y, u_numerical, u_exact, solution_type=3):
    """Визуализация результатов"""
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(15, 10))

    # Численное решение
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_surface(X, Y, u_numerical, cmap=cm.viridis)
    ax1.set_title('Численное решение')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')

    # Точное решение
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(X, Y, u_exact, cmap=cm.plasma)
    ax2.set_title('Точное решение')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')

    # Ошибка
    ax3 = fig.add_subplot(223, projection='3d')
    error = np.abs(u_numerical - u_exact)
    ax3.plot_surface(X, Y, error, cmap=cm.hot)
    ax3.set_title('Абсолютная ошибка')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Ошибка')

    # Срез по середине области
    # ax4 = fig.add_subplot(224)
    # mid_idx = len(y) // 2
    # ax4.plot(x, u_numerical[mid_idx, :], 'b-', label='Численное')
    # ax4.plot(x, u_exact[mid_idx, :], 'r--', label='Точное')
    # ax4.set_title(f'Срез при y = {y[mid_idx]:.2f}')
    # ax4.legend()
    # ax4.grid(True)

    plt.suptitle(f'Решение уравнения теплопроводности (Вариант {solution_type})')
    plt.tight_layout()
    return fig


nx, ny = 10, 10
T = 1.0
nt = 4000

# Решение задачи
x, y, u_num, u_exact, error_list = fractional_step_method(solution_type=3, nx=nx, ny=ny, T=T, nt=nt)



# Проверка граничных условий
def check_boundaries(u, x, y, t, solution_type):
    errors = []
    # Проверка границ по x (j=0 и j=nx)
    for i in range(ny + 1):
        computed = u[i, 0]
        exact = exact_solution(x[0], y[i], t, solution_type)
        errors.append(abs(computed - exact))

        computed = u[i, -1]
        exact = exact_solution(x[-1], y[i], t, solution_type)
        errors.append(abs(computed - exact))

    # Проверка границ по y (i=0 и i=ny)
    for j in range(nx + 1):
        computed = u[0, j]
        exact = exact_solution(x[j], y[0], t, solution_type)
        errors.append(abs(computed - exact))

        computed = u[-1, j]
        exact = exact_solution(x[j], y[-1], t, solution_type)
        errors.append(abs(computed - exact))

    return max(errors)


max_boundary_error = check_boundaries(u_num, x, y, T, 3)
print(f"Максимальная ошибка на границе: {max_boundary_error:.2e} (~{max_boundary_error:.5f})")

# Вычисление ошибок
l2_err, max_err = calculate_error(u_num, u_exact)
print(f"Ошибка L2: {l2_err:.2e} (~{l2_err:.5f})")
print(f"Максимальная ошибка: {max_err:.2e} (~{max_err:.5f})")

# Вывод ошибок по времени
print("\nОшибки L2 через каждые 100 временных шагов:")
for i, (t, err) in enumerate(error_list):
    if i % 100 == 0:
        print(f"t = {t:.5f}, L2-ошибка = {err:.5e}")

# Визуализация
fig = plot_solution(x, y, u_num, u_exact)
plt.show()