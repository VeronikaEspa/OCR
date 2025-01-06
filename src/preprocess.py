# preprocess.py
import numpy as np
from PIL import Image
from config import IMAGE_SIZE

def preprocess_image(image_path, image_size=IMAGE_SIZE):
    """
    Preprocesar una imagen desde un archivo.
    """
    img = Image.open(image_path).convert('L')
    img = img.resize(image_size)
    img = np.array(img).astype('float32') / 255.0  # Normalizar
    img = np.expand_dims(img, axis=-1)  # Canal extra
    img = np.expand_dims(img, axis=0)   # Dimensión batch
    return img

def preprocess_image_from_pil(img, image_size=IMAGE_SIZE):
    """
    Preprocesar una imagen desde una instancia de PIL.Image.
    """
    img = img.convert('L')
    img = img.resize(image_size)
    img = np.array(img).astype('float32') / 255.0  # Normalizar
    img = np.expand_dims(img, axis=-1)  # Canal extra
    img = np.expand_dims(img, axis=0)   # Dimensión batch
    return img

def preprocess_data(data_path):
    """
    Cargar y preprocesar los datos.
    """
    data = np.load(data_path)
    X, y = data['images'], data['labels']
    
    print(f"Forma original de X: {X.shape}")
    
    preprocessed_images = []
    valid_labels = []
    
    for idx, img in enumerate(X):
        try:
            # Verificar la dimensión de la imagen
            if img.ndim == 2:
                # Imagen en escala de grises
                pass
            elif img.ndim == 3 and img.shape[2] == 1:
                # Imagen con un canal
                img = img.squeeze(axis=2)
            else:
                print(f"Imagen {idx} tiene una forma inesperada: {img.shape}. Será omitida.")
                continue  # O manejar de otra manera

            # Verificar el tamaño de la imagen
            if img.shape[0] < 10 or img.shape[1] < 10:
                print(f"Imagen {idx} tiene un tamaño inválido: {img.shape}. Será omitida.")
                continue  # O manejar de otra manera

            # Convertir a uint8 si no lo está
            if img.dtype != 'uint8':
                img = img.astype('uint8')

            # Convertir a PIL Image y redimensionar
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(IMAGE_SIZE)
            img_resized = np.array(pil_img)

            # Asegurarse de que la imagen redimensionada tiene el tamaño correcto
            if img_resized.shape != IMAGE_SIZE:
                print(f"Imagen {idx} no se pudo redimensionar correctamente: {img_resized.shape}. Será omitida.")
                continue

            preprocessed_images.append(img_resized)
            valid_labels.append(y[idx])
        
        except Exception as e:
            print(f"Error procesando la imagen {idx}: {e}. Será omitida.")
            continue  # O manejar de otra manera

    # Convertir listas a arrays de NumPy
    X = np.array(preprocessed_images)
    y = np.array(valid_labels)
    
    print(f"Cantidad de imágenes válidas: {X.shape[0]}")
    print(f"Forma preprocesada de X: {X.shape}")
    
    # Normalizar
    X = X.astype('float32') / 255.0  # Normalizar
    X = np.expand_dims(X, axis=-1)
    
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    num_classes = y.shape[1]  # Determinar el número de clases

    return X_train, X_test, y_train, y_test, num_classes