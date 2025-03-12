import numpy as np
import matplotlib.pyplot as plt


# Задание производной
def derivative(x, y):
    return y / (x + 1) - y ** 2


# Точное решение
def exact_solution(x):
    return 2 * (x + 1) / (x ** 2 + 2 * x + 2)


# Метод Адамса второй степени
def adams_second_order_method(f, x0, y0, h, x_end):
    x_values = np.arange(x0, x_end + h, h)
    y_values = np.zeros(len(x_values))
    y_values[0] = y0

    # Первый шаг методом Эйлера
    y_values[1] = y0 + h * f(x0, y0)

    # Основной цикл метода Адамса второй степени
    for n in range(1, len(x_values) - 1):
        x_n1 = x_values[n]
        y_n1 = y_values[n]

        f_n1 = f(x_n1, y_n1)

        # Адамс второго порядка
        predictor = y_n1 + h * f_n1
        y_values[n + 1] = y_n1 + h / 2 * (f_n1 + f(x_values[n + 1], predictor))

    return x_values, y_values


# Параметры задачи
x0 = 0
x_end = 1
y0 = 1

# Шаги
h1 = 0.1
h2 = h1 / 2

# Численное решение
x_values1, y_values1 = adams_second_order_method(derivative, x0, y0, h1, x_end)
x_values2, y_values2 = adams_second_order_method(derivative, x0, y0, h2, x_end)

# Точное решение
exact_values1 = exact_solution(x_values1)
exact_values2 = exact_solution(x_values2)

# Погрешность
error1 = np.abs(y_values1 - exact_values1)
error2 = np.abs(y_values2 - exact_values2)

# Табличка со значениями шагов и погрешности
print("Таблица значений для шага h =", h1)
print("x\t\tЧисленное значение\tТочное значение\t\tПогрешность")
for x, y_num, y_exact, err in zip(x_values1, y_values1, exact_values1, error1):
    print(f"{x:.2f}\t{y_num:.6f}\t\t{y_exact:.6f}\t\t{err:.6f}")

print("\nТаблица значений для шага h =", h2)
print("x\t\tЧисленное значение\tТочное значение\t\tПогрешность")
for x, y_num, y_exact, err in zip(x_values2, y_values2, exact_values2, error2):
    print(f"{x:.2f}\t{y_num:.6f}\t\t{y_exact:.6f}\t\t{err:.6f}")
