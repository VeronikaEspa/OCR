# train_model.py

from config import MODEL_WEIGHTS, HISTORY_PATH, DATA_PATH, DATA_ZIP_PATH, EPOCHS
from model import build_model
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import zipfile
from io import BytesIO
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_npz_data(npz_path, fraction=0.2):
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
        # Para versiones más antiguas de Pillow
        resampling_filter = Image.LANCZOS
    return resampling_filter

def load_zip_data(zip_path, fraction=1.0, target_size=(28, 28)):
    """
    Carga una fracción de los datos desde un archivo .zip, redimensionando las imágenes al tamaño objetivo.

    Args:
        zip_path (str): Ruta al archivo .zip.
        fraction (float): Fracción de datos a cargar.
        target_size (tuple): Tamaño al que redimensionar las imágenes (alto, ancho).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Imágenes y etiquetas seleccionadas.
    """
    images = []
    labels = []
    resampling_filter = get_resampling_filter()
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Listar todos los archivos en el zip
            all_files = zip_ref.namelist()

            # Filtrar los archivos que están dentro de 'dataset1/' y no son directorios
            dataset_files = [f for f in all_files if f.startswith('dataset1/') and not f.endswith('/')]

            if not dataset_files:
                print("No se encontraron archivos dentro de 'dataset1/' en el zip.")
                return np.array([]), np.array([])

            # Extraer las etiquetas de los nombres de las carpetas
            label_to_files = {}
            for file in dataset_files:
                parts = file.split('/')
                if len(parts) < 3:
                    continue  # Esperamos 'dataset1/etiqueta/imagen'
                label = parts[1]
                label_to_files.setdefault(label, []).append(file)

            # Mapear etiquetas de letras a índices numéricos (0 -> 'A', 1 -> 'B', ..., 25 -> 'Z', 26 -> 'a', ..., 51 -> 'z')
            # Primero, identificar las etiquetas mayúsculas y minúsculas
            uppercase_labels = sorted([label for label in label_to_files.keys() if label.isupper()])
            lowercase_labels = sorted([label for label in label_to_files.keys() if label.islower()])

            # Asignar índices: mayúsculas 0-25, minúsculas 26-51
            label_to_index = {}
            for idx, label in enumerate(uppercase_labels):
                label_to_index[label] = idx  # 'A'->0, 'B'->1, ..., 'Z'->25
            for idx, label in enumerate(lowercase_labels):
                label_to_index[label] = idx + 26  # 'a'->26, 'b'->27, ..., 'z'->51

            # Seleccionar aleatoriamente el 50% de los datos por clase
            for label, files in label_to_files.items():
                num_select = max(1, int(len(files) * fraction))  # Asegurar al menos 1 muestra
                selected_files = random.sample(files, num_select)

                for file in selected_files:
                    with zip_ref.open(file) as image_file:
                        image_data = image_file.read()
                        image = Image.open(BytesIO(image_data))
                        image = image.convert('L')  # Convertir a escala de grises
                        image = image.resize(target_size, resampling_filter)  # Redimensionar
                        image_np = np.array(image)
                        images.append(image_np)
                        labels.append(label_to_index[label])

    except FileNotFoundError:
        print(f"El archivo {zip_path} no se encontró.")
        return np.array([]), np.array([])
    except zipfile.BadZipFile:
        print(f"El archivo {zip_path} no es un archivo zip válido.")
        return np.array([]), np.array([])

    X = np.array(images)
    y = np.array(labels)

    print(f"Cargado {X.shape[0]} muestras del archivo .zip.")

    return X, y

# def preprocess_and_split_data(npz_X, npz_y, zip_X, zip_y):
#     """
#     Divide los datos cargados en conjuntos de entrenamiento y prueba.

#     Args:
#         npz_X (np.ndarray): Imágenes del archivo .npz.
#         npz_y (np.ndarray): Etiquetas del archivo .npz.
#         zip_X (np.ndarray): Imágenes del archivo .zip.
#         zip_y (np.ndarray): Etiquetas del archivo .zip.

#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
#             X_train, X_test, y_train, y_test, num_classes
#     """
#     # Concatenar los conjuntos de entrenamiento
#     X_train = np.concatenate((npz_X, zip_X), axis=0)
#     y_train = np.concatenate((npz_y, zip_y), axis=0)

#     # Dividir el conjunto combinado en entrenamiento y prueba
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
#     )

#     # Obtener el número de clases
#     num_classes = len(np.unique(y_train))

#     print(f"Total de entrenamiento: {X_train.shape[0]} muestras.")
#     print(f"Total de prueba: {X_test.shape[0]} muestras.")
#     print(f"Número de clases: {num_classes}.")  # Debería ser 52

#     return X_train, X_test, y_train, y_test, num_classes


def preprocess_and_split_data(npz_X, npz_y, zip_X, zip_y):
    """
    Divide los datos cargados en conjuntos de entrenamiento y prueba.
    Asegura que las clases estén equilibradas entre mayúsculas y minúsculas.

    Args:
        npz_X (np.ndarray): Imágenes del archivo .npz.
        npz_y (np.ndarray): Etiquetas del archivo .npz.
        zip_X (np.ndarray): Imágenes del archivo .zip.
        zip_y (np.ndarray): Etiquetas del archivo .zip.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
            X_train, X_test, y_train, y_test, num_classes
    """
    # Concatenar los conjuntos de datos
    X_data = np.concatenate((npz_X, zip_X), axis=0)
    y_data = np.concatenate((npz_y, zip_y), axis=0)

    # Verificar distribución de clases
    unique_classes, class_counts = np.unique(y_data, return_counts=True)
    print("\nDistribución inicial de clases:")
    for cls, count in zip(unique_classes, class_counts):
        class_name = chr(65 + cls) if cls < 26 else chr(97 + (cls - 26))
        print(f"Clase {class_name}: {count} imágenes")

    # Calcular la cantidad mínima de muestras por clase
    min_samples_per_class = min(class_counts)

    # Balancear las clases tomando una cantidad igual de muestras para cada clase
    balanced_X = []
    balanced_y = []
    for cls in unique_classes:
        cls_indices = np.where(y_data == cls)[0]
        sampled_indices = np.random.choice(cls_indices, size=min_samples_per_class, replace=False)
        balanced_X.append(X_data[sampled_indices])
        balanced_y.append(y_data[sampled_indices])

    # Concatenar las clases balanceadas
    balanced_X = np.concatenate(balanced_X, axis=0)
    balanced_y = np.concatenate(balanced_y, axis=0)

    print(f"\nDatos balanceados: {balanced_X.shape[0]} imágenes en total.")
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_X, balanced_y, test_size=0.3, random_state=42, stratify=balanced_y
    )

    # Verificar la distribución en el conjunto de entrenamiento
    unique_classes_train, class_counts_train = np.unique(y_train, return_counts=True)
    print("\nDistribución de clases en el conjunto de entrenamiento:")
    for cls, count in zip(unique_classes_train, class_counts_train):
        class_name = chr(65 + cls) if cls < 26 else chr(97 + (cls - 26))
        print(f"Clase {class_name}: {count} imágenes")

    # Obtener el número de clases
    num_classes = len(unique_classes)

    print(f"\nTotal de entrenamiento: {X_train.shape[0]} muestras.")
    print(f"Total de prueba: {X_test.shape[0]} muestras.")
    print(f"Número de clases: {num_classes}.")  # Debería ser 52

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
    # Normalizar las imágenes
    X_train_prep = X_train.astype('float32') / 255.0
    X_test_prep = X_test.astype('float32') / 255.0

    # Expandir dimensiones si es necesario (ejemplo: agregar canal)
    if X_train_prep.ndim == 3:
        X_train_prep = np.expand_dims(X_train_prep, axis=-1)
        X_test_prep = np.expand_dims(X_test_prep, axis=-1)

    # Convertir etiquetas a one-hot y asegurar tipo float32
    y_train_prep = np.eye(num_classes, dtype='float32')[y_train]
    y_test_prep = np.eye(num_classes, dtype='float32')[y_test]

    return X_train_prep, X_test_prep, y_train_prep, y_test_prep, num_classes

def visualize_samples(X_train, y_train, X_zip, y_zip, label_to_class, num_samples=5):
    """
    Visualiza algunas muestras de ambos conjuntos de datos.

    Args:
        X_train (np.ndarray): Imágenes de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento (one-hot).
        X_zip (np.ndarray): Imágenes del archivo .zip.
        y_zip (np.ndarray): Etiquetas del archivo .zip.
        label_to_class (dict): Mapeo de etiquetas a nombres de clase.
        num_samples (int): Número de muestras a visualizar por conjunto.
    """
    plt.figure(figsize=(10, 4))

    # Muestras de entrenamiento (.npz)
    for i in range(num_samples):
        idx = random.randint(0, X_train.shape[0] - 1)
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X_train[idx].squeeze(), cmap='gray')
        label = np.argmax(y_train[idx])
        plt.title(f"Clase: {label_to_class[label]}")
        plt.axis('off')

    # Muestras del archivo .zip
    for i in range(num_samples):
        idx = random.randint(0, X_zip.shape[0] - 1)
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(X_zip[idx].squeeze(), cmap='gray')
        # Asumiendo que y_zip son etiquetas numéricas ya mapeadas a 26-51
        label = y_zip[idx]
        plt.title(f"Clase: {label_to_class[label]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def train_ocr_model():
    # Crear el directorio 'models/' si no existe
    os.makedirs("models/", exist_ok=True)

    # Cargar datos del archivo .npz
    npz_X, npz_y = load_npz_data(DATA_PATH, fraction=0.5)

    # Cargar datos del archivo .zip
    zip_X, zip_y = load_zip_data(DATA_ZIP_PATH, fraction=0.5, target_size=(28, 28))

    # Verificar que se hayan cargado datos de ambos orígenes
    if npz_X.size == 0 or zip_X.size == 0:
        print("No se pudieron cargar datos de uno o ambos archivos. Abortando entrenamiento.")
        return

    # Preprocesar y dividir los datos
    X_train, X_test, y_train, y_test, num_classes = preprocess_and_split_data(npz_X, npz_y, zip_X, zip_y)

    # Preprocesar los datos
    X_train_prep, X_test_prep, y_train_prep, y_test_prep, num_classes = preprocess_data(
        X_train, X_test, y_train, y_test, num_classes
    )

    # Mapeo de etiquetas para visualización
    # Suponiendo que los índices 0-25 son 'A'-'Z' y 26-51 son 'a'-'z'
    label_to_class = {}
    for label in range(num_classes):
        if label < 26:
            class_name = chr(65 + label)  # 'A' = 65 en ASCII
        else:
            class_name = chr(97 + (label - 26))  # 'a' = 97 en ASCII
        label_to_class[label] = class_name

    # Visualizar algunas muestras para verificar
    visualize_samples(X_train_prep, y_train_prep, zip_X, zip_y, label_to_class, num_samples=5)

    # Construir el modelo
    model = build_model(X_train_prep.shape[1:], num_classes)
    
    # Imprimir el resumen del modelo para verificar las dimensiones
    model.summary()

    # Definir el optimizador con una tasa de aprendizaje reducida
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Definir el generador de aumento de datos
    datagen = ImageDataGenerator(
        rotation_range=10,       # Rotaciones aleatorias en el rango de 10 grados
        width_shift_range=0.1,   # Desplazamiento horizontal
        height_shift_range=0.1,  # Desplazamiento vertical
        zoom_range=0.1,          # Zoom aleatorio
        shear_range=0.1,         # Cizallamiento
        horizontal_flip=False,   # No voltear horizontalmente
        vertical_flip=False      # No voltear verticalmente
    )

    # Ajustar el generador a los datos de entrenamiento
    datagen.fit(X_train_prep)

    # Calcular los pesos de las clases para manejar el desequilibrio
    y_integers = np.argmax(y_train_prep, axis=1)
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weights = dict(enumerate(class_weights_array))

    # Definir callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath="models/ocr_model_best.weights.h5",  # Cambiar a .weights.h5
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        save_weights_only=True  # Mantener que solo se guarden los pesos
        # save_format='h5'  # Este argumento ha sido eliminado
    )

    # Entrenar el modelo utilizando el generador con callbacks y class_weights
    history = model.fit(
        datagen.flow(X_train_prep, y_train_prep, batch_size=64),
        steps_per_epoch=len(X_train_prep),
        validation_data=(X_test_prep, y_test_prep),
        epochs=EPOCHS,
        callbacks=[reduce_lr, early_stop, checkpoint],
        # class_weight=class_weights
    )

    # Cargar los mejores pesos antes de guardar el modelo completo
    model.load_weights("models/ocr_model_best.weights.h5")  # Cambiar a .weights.h5
    model.save(MODEL_WEIGHTS)  # Guardar el modelo completo en .h5

    # Guardar el historial de entrenamiento
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    print("Modelo entrenado y guardado correctamente.")

if __name__ == "__main__":
    train_ocr_model()