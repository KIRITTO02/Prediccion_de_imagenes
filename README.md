# Prediccion_de_imagenes

1. mnist_cnn.py - Entrenamiento del Modelo de Reconocimiento de Dígitos
Descripción:
Este script implementa una red neuronal convolucional (CNN) para reconocer dígitos escritos a mano utilizando el conjunto de datos MNIST. El modelo se entrena, evalúa y guarda para su uso posterior.

Características principales:

Carga y preprocesa el dataset MNIST (60,000 imágenes de entrenamiento + 10,000 de prueba)

Define una arquitectura CNN con:

2 capas convolucionales con ReLU

Capas de Max Pooling

Dropout para regularización

Capa densa final con softmax

Entrena el modelo con:

Optimizador Adam

Función de pérdida categorical crossentropy

15 épocas de entrenamiento

Genera gráficos de:

Precisión (accuracy) durante el entrenamiento

Pérdida (loss) durante el entrenamiento

Evalúa el modelo con el conjunto de prueba

Guarda el modelo entrenado como mnist_cnn_model.h5

Incluye funcionalidad básica para probar con imágenes propias

Uso:

bash
python mnist_cnn.py
Salida esperada:

Gráficos de entrenamiento

Precisión final en el conjunto de prueba (~99%)

Modelo guardado en formato HDF5

2. probar_modelo.py - Prueba del Modelo con Imágenes Personalizadas
Descripción:
Este script carga el modelo entrenado y lo utiliza para hacer predicciones sobre imágenes personalizadas de dígitos escritos a mano.

Características principales:

Carga el modelo preentrenado (mnist_cnn_model.h5)

Procesa imágenes personalizadas:

Conversión a escala de grises

Redimensionamiento a 28x28 píxeles

Inversión de colores (para coincidir con formato MNIST)

Normalización de valores de píxeles

Muestra resultados con:

Visualización de la imagen original y procesada

Dígito predicho

Nivel de confianza de la predicción

Soporta múltiples formatos de imagen (PNG, JPG, JPEG)

Requisitos:

El modelo entrenado (mnist_cnn_model.h5) debe existir

Las imágenes deben estar en un directorio llamado ima/

Imágenes preferiblemente con:

Fondo claro (blanco)

Dígito oscuro (negro)

Formato cuadrado

Uso:

bash
python probar_modelo.py
Salida esperada:

Para cada imagen en ima/:

Muestra comparación imagen original/procesada

Imprime predicción y confianza

Ejemplo: Imagen: mi_5.png -> Dígito: 5 (Confianza: 98.72%)
