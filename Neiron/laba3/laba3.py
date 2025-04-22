import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Создание директории для графиков
os.makedirs("grafiki", exist_ok=True)

# Загрузка и предобработка данных
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Список для хранения результатов
results = []

# Функция построения модели
def build_model(use_dropout=False, use_bn=False, learning_rate=0.001):
    model = models.Sequential()
    model.add(layers.Input(shape=(784,)))

    model.add(layers.Dense(128))
    if use_bn:
        model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    if use_dropout:
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(64))
    if use_bn:
        model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    if use_dropout:
        model.add(layers.Dropout(0.3))

    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Функция для обучения и сохранения графиков
def train_and_plot(model, epochs, title='', filename='plot.png'):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=1, batch_size=64)

    # Графики
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучающая выборка')
    plt.plot(history.history['val_accuracy'], label='Тестовая выборка')
    plt.title(f'Точность: {title}')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучающая выборка')
    plt.plot(history.history['val_loss'], label='Тестовая выборка')
    plt.title(f'Потери: {title}')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"grafiki/{filename}", dpi=300)
    plt.close()

    # Сбор статистики
    final_train_acc = history.history['accuracy'][-1] * 100
    final_test_acc = history.history['val_accuracy'][-1] * 100
    final_train_loss = history.history['loss'][-1]
    final_test_loss = history.history['val_loss'][-1]

    results.append({
        'Название': title,
        'Точн. обуч. (%)': round(final_train_acc, 1),
        'Точн. теста (%)': round(final_test_acc, 1),
        'Потери обуч.': round(final_train_loss, 2),
        'Потери теста': round(final_test_loss, 2)
    })

# Эксперимент 1
print("\nЭксперимент: Влияние числа эпох")
train_and_plot(build_model(), 5, '5 эпох', 'epoh_5.png')
train_and_plot(build_model(), 10, '10 эпох', 'epoh_10.png')
train_and_plot(build_model(), 20, '20 эпох', 'epoh_20.png')
train_and_plot(build_model(), 30, '30 эпох', 'epoh_30.png')
train_and_plot(build_model(), 50, '50 эпох', 'epoh_50.png')

# Эксперимент 2
print("\nЭксперимент: Learning Rate")
train_and_plot(build_model(learning_rate=0.0001), 30, 'LR = 0.0001', 'lr_0001.png')
train_and_plot(build_model(learning_rate=0.01), 30, 'LR = 0.01', 'lr_01.png')

# Эксперимент 3
print("\nЭксперимент: Dropout")
train_and_plot(build_model(use_dropout=False), 30, 'Без Dropout', 'dropout_off.png')
train_and_plot(build_model(use_dropout=True), 30, 'С Dropout', 'dropout_on.png')

# Эксперимент 4
print("\nЭксперимент: BatchNorm")
train_and_plot(build_model(use_bn=False), 30, 'Без BatchNorm', 'bn_off.png')
train_and_plot(build_model(use_bn=True), 30, 'С BatchNorm', 'bn_on.png')

print("\nИтоговая таблица результатов:")
print("{:<25} {:>15} {:>15} {:>15} {:>15}".format("Эксперимент", "Точн. обуч. (%)", "Точн. теста (%)", "Потери обуч.", "Потери теста"))
print("-" * 90)
for r in results:
    print("{:<25} {:>15} {:>15} {:>15} {:>15}".format(
        r['Название'],
        r['Точн. обуч. (%)'],
        r['Точн. теста (%)'],
        r['Потери обуч.'],
        r['Потери теста']
    ))
