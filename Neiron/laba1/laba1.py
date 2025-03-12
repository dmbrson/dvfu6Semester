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

# Создаем модель
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
model.fit(train_images, train_labels, epochs=5, batch_size=32, validation_data=(test_images, test_labels))

# Оцениваем качество модели
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Точность на тестовых данных: {test_acc * 100:.2f}%')


show_predictions(10)