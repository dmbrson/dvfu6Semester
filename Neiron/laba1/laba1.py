import numpy as np
import struct
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def show_predictions(num_samples=10):
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(test_images[idx], cmap='gray')
        prediction = model.predict(np.expand_dims(test_images[idx], axis=0), verbose=0)
        ax.set_title(f'Предсказано: {np.argmax(prediction)}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        _, num, rows, cols = struct.unpack('>IIII', f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols).astype(np.float32) / 255.0

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        _, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# Загружаем данные
train_images = load_mnist_images('train-images.idx3-ubyte')
train_labels = load_mnist_labels('train-labels.idx1-ubyte')
test_images = load_mnist_images('t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Преобразуем 28x28 вектор в одномерный массив (784)
    layers.Dense(128, activation='relu'),  # Первый скрытый слой с 128 нейронами
    layers.Dense(10, activation='softmax') # Выходной слой с 10 нейронами (по числу классов)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))

# Оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Точность на тестовых данных: {test_acc * 100:.2f}%')

# Вывод весов и смещений для анализа параметров модели
for layer in model.layers:
    if isinstance(layer, layers.Dense):
        weights, biases = layer.get_weights()
        print(f'Слой: {layer.name}, Форма весов: {weights.shape}, Форма смещений: {biases.shape}')
        print(f'Слой: {layer.name}')
        print(f'Первые 10 весов: {weights.flatten()[:10]}')
        print(f'Первые 10 смещений: {biases[:10]}\n')

# Визуализация функции потерь во время обучения
plt.plot(history.history['loss'], label='Функция потерь (Train)')
plt.plot(history.history['val_loss'], label='Функция потерь (Validation)')
plt.xlabel('Эпохи')
plt.ylabel('Значение функции потерь')
plt.legend()
plt.show()

# Визуализация точности во время обучения
plt.plot(history.history['accuracy'], label='Точность (Train)')
plt.plot(history.history['val_accuracy'], label='Точность (Validation)')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

# Показываем предсказания
show_predictions(10)