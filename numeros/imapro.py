import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Cargar el modelo guardado
model = tf.keras.models.load_model('mnist_cnn_model.h5')

def load_and_predict(image_path):
    """
    Carga una imagen personalizada, la preprocesa y hace una predicción
    """
    try:
        # Cargar y preprocesar la imagen
        img = Image.open(image_path).convert('L')  # Convertir a escala de grises
        img = img.resize((28, 28))                # Redimensionar
        
        # Mostrar imagen original
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Imagen original")
        plt.axis('off')
        
        # Preprocesamiento (invertir colores como en MNIST)
        img = np.array(img)
        img = 255 - img  # Invertir colores
        
        # Normalizar y preparar para el modelo
        img = img.astype("float32") / 255.0
        img_processed = np.expand_dims(img, axis=-1)  # Añadir canal
        img_processed = np.expand_dims(img_processed, axis=0)  # Añadir batch
        
        # Hacer predicción
        prediction = model.predict(img_processed)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Mostrar imagen procesada y resultados
        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {predicted_digit}\nConf: {confidence:.2%}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return predicted_digit, confidence
    
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None, None

# Probar con imágenes
if __name__ == "__main__":
    # Directorio con imágenes (debe estar en el mismo lugar que este script)
    images_dir = "ima"
    
    if os.path.exists(images_dir):
        print("Probando con imágenes propias...")
        for img_file in sorted(os.listdir(images_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, img_file)
                digit, confidence = load_and_predict(img_path)
                print(f"Imagen: {img_file} -> Dígito: {digit} (Confianza: {confidence:.2%})")
    else:
        print(f"ERROR: Crea un directorio llamado '{images_dir}' y coloca allí tus imágenes.")
        print("Las imágenes deben ser cuadradas preferiblemente, con el dígito en negro y fondo blanco.")