# train_model.py

from config import MODEL_WEIGHTS, HISTORY_PATH, DATA_PATH, DATA_ZIP_PATH, EPOCHS, MODEL_WEIGHTS_BEST
from model import build_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import zipfile
from io import BytesIO
from PIL import Image
import random
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from performance import medir_tiempo


def load_npz_data(npz_path, fraction=1.0):
    """
    Carga una fracción de los datos desde un archivo .npz.

    Args:
        npz_path (str): Ruta al archivo .npz.
        fraction (float): Fracción de datos a cargar.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Imágenes y etiquetas seleccionadas.
    """
    try:
        data = np.load(npz_path)
        X = data['images']
        y = data['labels']
    except FileNotFoundError:
        print(f"El archivo {npz_path} no se encontró.")
        return np.array([]), np.array([])
    except KeyError:
        print(f"El archivo {npz_path} no contiene las claves 'images' y 'labels'.")
        return np.array([]), np.array([])

    total_samples = X.shape[0]
    selected_indices = np.random.choice(total_samples, size=int(total_samples * fraction), replace=False)
    X_selected = X[selected_indices]
    y_selected = y[selected_indices]

    print(f"Cargado {X_selected.shape[0]} muestras del archivo .npz.")
    return X_selected, y_selected


def get_resampling_filter():
    """
    Obtiene el filtro de remuestreo adecuado según la versión de Pillow instalada.

    Returns:
        PIL.Image.Resampling: Filtro de remuestreo.
    """
    try:
        resampling_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resampling_filter = Image.LANCZOS
    return resampling_filter


def load_zip_data(zip_path, fraction=1.0, target_size=(28, 28)):
    """
    Carga una fracción de las imágenes desde un archivo .zip, redimensionándolas al tamaño objetivo.

    Args:
        zip_path (str): Ruta al archivo .zip.
        fraction (float): Fracción de datos a cargar (0.0 a 1.0).
        target_size (tuple): Tamaño al que redimensionar las imágenes (alto, ancho).

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: Imágenes, etiquetas seleccionadas y mapeo de etiquetas.
    """
    images = []
    labels = []
    label_to_index = {}
    resampling_filter = get_resampling_filter()

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            dataset_files = [f for f in all_files if f.startswith('dataset1/') and not f.endswith('/')]

            if not dataset_files:
                print("No se encontraron archivos dentro de 'dataset1/' en el zip.")
                return np.array([]), np.array([]), {}

            for file in dataset_files:
                parts = file.split('/')
                if len(parts) < 3:
                    continue
                label = parts[1]
                if label not in label_to_index:
                    label_to_index[label] = len(label_to_index)

                label_images = [f for f in dataset_files if f.startswith(f'dataset1/{label}/')]
                num_files = len(label_images)
                num_to_select = int(num_files * fraction)
                selected_files = random.sample(label_images, num_to_select) if fraction < 1.0 else label_images

                for selected_file in selected_files:
                    with zip_ref.open(selected_file) as image_file:
                        image_data = image_file.read()
                        image = Image.open(BytesIO(image_data)).convert('L')
                        image = image.resize(target_size, resampling_filter)
                        images.append(np.array(image))
                        labels.append(label_to_index[label])

    except FileNotFoundError:
        print(f"El archivo {zip_path} no se encontró.")
        return np.array([]), np.array([]), {}
    except zipfile.BadZipFile:
        print(f"El archivo {zip_path} no es un archivo zip válido.")
        return np.array([]), np.array([]), {}

    X = np.array(images)
    y = np.array(labels)

    print(f"Cargado {X.shape[0]} imágenes del archivo .zip.")
    print("Clases encontradas:")
    for label, index in label_to_index.items():
        print(f"Clase '{label}': Índice {index}")

    return X, y, label_to_index


def preprocess_and_split_data(npz_X, npz_y, zip_X, zip_y, mnist_X=None, mnist_y=None):
    """
    Divide los datos cargados en conjuntos de entrenamiento y prueba.

    Args:
        npz_X (np.ndarray): Imágenes del archivo .npz.
        npz_y (np.ndarray): Etiquetas del archivo .npz.
        zip_X (np.ndarray): Imágenes del archivo .zip.
        zip_y (np.ndarray): Etiquetas del archivo .zip.
        mnist_X (np.ndarray, optional): Imágenes de MNIST.
        mnist_y (np.ndarray, optional): Etiquetas de MNIST.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
            X_train, X_test, y_train, y_test, num_classes
    """
    if mnist_X is not None and mnist_y is not None:
        X_data = np.concatenate((npz_X, zip_X, mnist_X), axis=0)
        y_data = np.concatenate((npz_y, zip_y, mnist_y), axis=0)
    else:
        X_data = np.concatenate((npz_X, zip_X), axis=0)
        y_data = np.concatenate((npz_y, zip_y), axis=0)

    unique_classes, class_counts = np.unique(y_data, return_counts=True)
    print("\nDistribución inicial de clases:")
    for cls, count in zip(unique_classes, class_counts):
        class_name = chr(48 + cls) if cls < 10 else f"Clase {cls}"
        print(f"Clase {class_name}: {count} imágenes")

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )

    unique_classes_train, class_counts_train = np.unique(y_train, return_counts=True)
    print("\nDistribución de clases en el conjunto de entrenamiento:")
    for cls, count in zip(unique_classes_train, class_counts_train):
        class_name = chr(48 + cls) if cls < 10 else f"Clase {cls}"
        print(f"Clase {class_name}: {count} imágenes")

    num_classes = len(unique_classes)

    print(f"\nTotal de entrenamiento: {X_train.shape[0]} muestras.")
    print(f"Total de prueba: {X_test.shape[0]} muestras.")
    print(f"Número de clases: {num_classes}.")

    return X_train, X_test, y_train, y_test, num_classes


def preprocess_data(X_train, X_test, y_train, y_test, num_classes):
    """
    Preprocesa los datos para el entrenamiento.

    Args:
        X_train (np.ndarray): Imágenes de entrenamiento.
        X_test (np.ndarray): Imágenes de prueba.
        y_train (np.ndarray): Etiquetas de entrenamiento.
        y_test (np.ndarray): Etiquetas de prueba.
        num_classes (int): Número de clases.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
            X_train_prep, X_test_prep, y_train_prep, y_test_prep, num_classes
    """
    X_train_prep = X_train.astype('float32') / 255.0
    X_test_prep = X_test.astype('float32') / 255.0

    if X_train_prep.ndim == 3:
        X_train_prep = np.expand_dims(X_train_prep, axis=-1)
        X_test_prep = np.expand_dims(X_test_prep, axis=-1)

    y_train_prep = np.eye(num_classes, dtype='float32')[y_train]
    y_test_prep = np.eye(num_classes, dtype='float32')[y_test]

    return X_train_prep, X_test_prep, y_train_prep, y_test_prep, num_classes


def load_mnist_data():
    """
    Carga y preprocesa el dataset MNIST.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Imágenes y etiquetas de MNIST.
    """
    print("\nCargando el dataset MNIST...")
    (mnist_X_train, mnist_y_train), (mnist_X_test, mnist_y_test) = mnist.load_data()
    mnist_X = np.concatenate((mnist_X_train, mnist_X_test), axis=0)
    mnist_y = np.concatenate((mnist_y_train, mnist_y_test), axis=0)
    print(f"MNIST cargado: {mnist_X.shape[0]} imágenes.")
    return mnist_X, mnist_y


def train_ocr_model():
    os.makedirs("models/", exist_ok=True)

    # Cargar datos desde .npz y .zip
    npz_X, npz_y = load_npz_data(DATA_PATH, fraction=0.1)
    zip_X, zip_y, zip_label_to_index = load_zip_data(DATA_ZIP_PATH, fraction=1.0, target_size=(28, 28))

    print("\nMapeo de etiquetas para el .zip:")
    for label, index in zip_label_to_index.items():
        print(f"Clase: {label} -> Índice: {index}")

    # Cargar datos de MNIST
    mnist_X, mnist_y = load_mnist_data()

    # Mapear etiquetas de MNIST si es necesario
    # Asumiendo que las etiquetas de MNIST son del 0 al 9 y no colisionan con otras etiquetas
    # Si hay colisión, ajusta las etiquetas de MNIST en consecuencia
    mnist_y_mapped = mnist_y  # Ajusta si es necesario

    if npz_X.size == 0 or zip_X.size == 0 or mnist_X.size == 0:
        print("No se pudieron cargar datos de uno o más archivos. Abortando entrenamiento.")
        return

    # Preprocesar y dividir los datos
    X_train, X_test, y_train, y_test, num_classes = preprocess_and_split_data(
        npz_X, npz_y, zip_X, zip_y, mnist_X=mnist_X, mnist_y=mnist_y_mapped
    )
    X_train_prep, X_test_prep, y_train_prep, y_test_prep, num_classes = preprocess_data(
        X_train, X_test, y_train, y_test, num_classes
    )

    # Construir el modelo
    model = build_model(X_train_prep.shape[1:], num_classes)
    model.summary()

    # Compilar el modelo
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Configurar el generador de datos
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    datagen.fit(X_train_prep)

    # Configurar callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath="models/ocr_model_best.weights.h5", monitor='val_accuracy',
        save_best_only=True, verbose=1, save_weights_only=True
    )

    # Entrenar el modelo
    history = model.fit(
        datagen.flow(X_train_prep, y_train_prep, batch_size=64),
        steps_per_epoch=len(X_train_prep) // 64,
        validation_data=(X_test_prep, y_test_prep),
        epochs=EPOCHS,
        callbacks=[reduce_lr, early_stop, checkpoint]
    )

    # Cargar los mejores pesos y guardar el modelo completo
    model.load_weights("models/ocr_model_best.weights.h5")
    model.save(MODEL_WEIGHTS)

    # Guardar el historial de entrenamiento
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    print("Modelo entrenado y guardado correctamente.")


train_ocr_model = medir_tiempo(train_ocr_model)

if __name__ == "__main__":
    train_ocr_model()