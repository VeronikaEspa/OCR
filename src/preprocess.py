import numpy as np
from PIL import Image, ImageOps
from config import IMAGE_SIZE
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def preprocess_image(image_input, image_size=IMAGE_SIZE):
    """
    Preprocesar una imagen desde una ruta de archivo o un objeto Image.

    Args:
        image_input (str or Image.Image): Ruta de la imagen a procesar o un objeto PIL Image.
        image_size (tuple): Tamaño al que se redimensionará la imagen.

    Returns:
        tuple: Imagen preprocesada lista para predicción y la imagen procesada para visualización.
    """
    try:
        if isinstance(image_input, str):
            with Image.open(image_input) as img:
                img = img.convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        else:
            raise ValueError("Entrada no válida para preprocess_image. Se esperaba una ruta o un objeto Image.")

        img_inverted = ImageOps.invert(img)

        img_array = np.array(img_inverted)
        threshold = 175

        for attempt in range(3):
            mask = img_array > threshold
            coords = np.column_stack(np.where(mask))

            if coords.size > 0:
                break
            threshold -= 25

        if coords.size == 0:
            print("Advertencia: No se detectaron píxeles claros, usando imagen original.")
            coords = np.array([[0, 0], [img_array.shape[0] - 1, img_array.shape[1] - 1]])


        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        margin = 6
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(img_array.shape[0], y_max + margin)
        x_max = min(img_array.shape[1], x_max + margin)

        cropped = img_inverted.crop((x_min, y_min, x_max + 1, y_max + 1))
        centered_img = cropped.resize(image_size, Image.BICUBIC)

        gray_centered_img = centered_img.convert("L")
        img_to_predict = np.array(gray_centered_img).astype("float32") / 255.0
        img_to_predict = np.expand_dims(np.expand_dims(img_to_predict, axis=-1), axis=0)

        image_to_display = np.squeeze(img_to_predict, axis=(0, -1))

        return img_to_predict, image_to_display

    except Exception as e:
        raise RuntimeError(f"Error durante el preprocesamiento de la imagen: {e}")

    except Exception as e:
        raise RuntimeError(f"Error durante el preprocesamiento de la imagen: {e}")


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
            if img.ndim == 2:
                pass
            elif img.ndim == 3 and img.shape[2] == 1:
                img = img.squeeze(axis=2)
            else:
                print(f"Imagen {idx} tiene una forma inesperada: {img.shape}. Será omitida.")
                continue

            if img.shape[0] < 10 or img.shape[1] < 10:
                print(f"Imagen {idx} tiene un tamaño inválido: {img.shape}. Será omitida.")
                continue

            if img.dtype != 'uint8':
                img = img.astype('uint8')

            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(IMAGE_SIZE)
            img_resized = np.array(pil_img)

            if img_resized.shape != IMAGE_SIZE:
                print(f"Imagen {idx} no se pudo redimensionar correctamente: {img_resized.shape}. Será omitida.")
                continue

            preprocessed_images.append(img_resized)
            valid_labels.append(y[idx])
        
        except Exception as e:
            print(f"Error procesando la imagen {idx}: {e}. Será omitida.")
            continue

    X = np.array(preprocessed_images)
    y = np.array(valid_labels)
    
    print(f"Cantidad de imágenes válidas: {X.shape[0]}")
    print(f"Forma preprocesada de X: {X.shape}")
    
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = to_categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    num_classes = y.shape[1]

    return X_train, X_test, y_train, y_test, num_classes