# Importar las bibliotecas necesarias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

# Cargar y preparar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxeles a [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Redimensionar las imágenes para incluir el canal (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Convertir las etiquetas a one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Mostrar información sobre los datos
print("Forma de x_train:", x_train.shape)
print("Forma de y_train:", y_train.shape)
print("Forma de x_test:", x_test.shape)
print("Forma de y_test:", y_test.shape)

# Visualizar algunos ejemplos del conjunto de entrenamiento
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.title(f"Etiqueta: {np.argmax(y_train[i])}")
    plt.axis("off")
plt.show()

# Crear el modelo de red neuronal convolucional (CNN)
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    
    # Primera capa convolucional
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Segunda capa convolucional
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Aplanar y capas densas
    layers.Flatten(),
    layers.Dropout(0.5),  # Regularización para evitar overfitting
    layers.Dense(10, activation="softmax"),
])

# Compilar el modelo
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1,  # Usar 10% de los datos para validación
)

# Evaluar el modelo con el conjunto de prueba
score = model.evaluate(x_test, y_test, verbose=0)
print(f"Pérdida en prueba: {score[0]}")
print(f"Precisión en prueba: {score[1]}")

# Graficar la precisión y pérdida durante el entrenamiento
plt.figure(figsize=(12, 5))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Precisión entrenamiento")
plt.plot(history.history["val_accuracy"], label="Precisión validación")
plt.title("Precisión durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Pérdida entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida validación")
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()

plt.tight_layout()
plt.show()

# Función para predecir y mostrar resultados
def predict_and_display(model, images, labels, num_images=5):
    # Seleccionar imágenes aleatorias
    indices = np.random.choice(range(len(images)), size=num_images)
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Hacer predicciones
    predictions = model.predict(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels, axis=1)
    
    # Mostrar resultados
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {predicted_classes[i]}\nReal: {true_classes[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Probar con imágenes del conjunto de prueba
predict_and_display(model, x_test, y_test)

# Guardar el modelo para uso futuro
model.save("mnist_cnn_model.h5")
print("Modelo guardado como mnist_cnn_model.h5")

# Código adicional para probar con imágenes propias (requiere preparación)
# Puedes cargar tus propias imágenes de dígitos y preprocesarlas para probar el modelo