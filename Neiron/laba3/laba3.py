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

# Эксперимент 1
print("\nЭксперимент 1: Влияние числа эпох")
train_and_plot(build_model(), 5, '5 эпох', 'exp1_5_epoh.png')
train_and_plot(build_model(), 30, '30 эпох', 'exp1_30_epoh.png')

# Эксперимент 2
print("\nЭксперимент 2: Влияние learning rate")
train_and_plot(build_model(learning_rate=0.0001), 5, 'LR=0.0001 (5 эпох)', 'exp2_lr0001_5epoh.png')
train_and_plot(build_model(learning_rate=0.0001), 30, 'LR=0.0001 (30 эпох)', 'exp2_lr0001_30epoh.png')
train_and_plot(build_model(learning_rate=0.01), 5, 'LR=0.01 (5 эпох)', 'exp2_lr01_5epoh.png')
train_and_plot(build_model(learning_rate=0.01), 30, 'LR=0.01 (30 эпох)', 'exp2_lr01_30epoh.png')

# Эксперимент 3
print("\nЭксперимент 3: Влияние Dropout")
train_and_plot(build_model(use_dropout=True), 5, 'С Dropout (5 эпох)', 'exp3_dropout_5epoh.png')
train_and_plot(build_model(use_dropout=True), 30, 'С Dropout (30 эпох)', 'exp3_dropout_30epoh.png')
train_and_plot(build_model(use_dropout=False), 5, 'Без Dropout (5 эпох)', 'exp3_nodropout_5epoh.png')
train_and_plot(build_model(use_dropout=False), 30, 'Без Dropout (30 эпох)', 'exp3_nodropout_30epoh.png')

# Эксперимент 4
print("\nЭксперимент 4: Влияние Batch Normalization")
train_and_plot(build_model(use_bn=True), 5, 'С BatchNorm (5 эпох)', 'exp4_bn_5epoh.png')
train_and_plot(build_model(use_bn=True), 30, 'С BatchNorm (30 эпох)', 'exp4_bn_30epoh.png')
train_and_plot(build_model(use_bn=False), 5, 'Без BatchNorm (5 эпох)', 'exp4_nobn_5epoh.png')
train_and_plot(build_model(use_bn=False), 30, 'Без BatchNorm (30 эпох)', 'exp4_nobn_30epoh.png')
