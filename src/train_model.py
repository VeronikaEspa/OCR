import os
from PIL import Image, ImageOps
import zipfile
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from model import build_model
from config import DATA_ZIP_PATH, DATA_EMNIST_PATH, DATA_PATH, MODEL_WEIGHTS, HISTORY_PATH, IMAGE_SIZE, EPOCHS

def load_and_merge_datasets():
    print("Cargando y uniendo datasets MNIST, EMNIST, ZIP y NPZ...")

    print("Cargando MNIST...")
    (mnist_images_train, mnist_labels_train), (mnist_images_test, mnist_labels_test) = mnist.load_data()
    mnist_images = np.concatenate([mnist_images_train, mnist_images_test])
    mnist_labels = [str(label) for label in np.concatenate([mnist_labels_train, mnist_labels_test])]
    mnist_images = np.array([np.expand_dims(np.array(Image.fromarray(img).resize(IMAGE_SIZE, Image.BICUBIC)), axis=-1)
                             for img in mnist_images])

    print("Cargando EMNIST...")
    emnist_images, emnist_labels = [], []
    with zipfile.ZipFile(DATA_EMNIST_PATH, 'r') as archive:
        for file_name in archive.namelist():
            if "emnist-letters-train-images" in file_name:
                emnist_images = np.frombuffer(archive.read(file_name), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
            elif "emnist-letters-train-labels" in file_name:
                emnist_labels = np.frombuffer(archive.read(file_name), dtype=np.uint8, offset=8)
    emnist_labels = [chr(label + 96) for label in emnist_labels]
    emnist_images = np.array([np.expand_dims(np.array(Image.fromarray(img).resize(IMAGE_SIZE, Image.BICUBIC)), axis=-1)
                               for img in emnist_images])

    print("Cargando ZIP...")
    zip_images, zip_labels = [], []
    with zipfile.ZipFile(DATA_ZIP_PATH, 'r') as archive:
        for file_name in archive.namelist():
            if file_name.endswith(('png', 'jpg', 'jpeg')):
                with archive.open(file_name) as img_file:
                    img = Image.open(img_file).convert('L')
                    img = ImageOps.invert(img)
                    img = img.resize(IMAGE_SIZE, Image.BICUBIC)
                    img_array = np.expand_dims(np.array(img), axis=-1)
                    zip_images.append(img_array)
                    label = os.path.dirname(file_name).split('/')[-1].strip()
                    zip_labels.append(label)
    zip_images = np.array(zip_images)

    print("Cargando NPZ...")
    npz_data = np.load(DATA_PATH)
    npz_images = np.array([np.expand_dims(np.array(Image.fromarray(img).resize(IMAGE_SIZE, Image.BICUBIC)), axis=-1)
                           for img in npz_data['images']])
    npz_labels = [chr(label + 65) for label in npz_data['labels']]  # Convertir etiquetas a mayúsculas

    print("Concatenando datasets...")
    images = np.concatenate([mnist_images, emnist_images, zip_images, npz_images])
    labels = np.concatenate([mnist_labels, emnist_labels, zip_labels, npz_labels])

    unique_labels = sorted(set(labels))
    print(f"Total clases únicas detectadas: {len(unique_labels)}")
    print(f"Etiquetas: {unique_labels}")

    print(f"Total imágenes: {images.shape[0]}, Total etiquetas: {len(labels)}")
    return images, labels


def preprocess_and_split(images, labels):
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y_indices = np.array([label_to_index[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(images, y_indices, test_size=0.3, random_state=42)

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    num_classes = len(label_to_index)
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)

    return X_train, X_test, y_train, y_test, num_classes

def train_model():
    os.makedirs("models/", exist_ok=True)

    images, labels = load_and_merge_datasets()
    X_train, X_test, y_train, y_test, num_classes = preprocess_and_split(images, labels)

    print(f"Total de clases: {num_classes}")
    print(f"Total de datos: {len(images)}")
    print(f"Datos de entrenamiento: {len(X_train)}")
    print(f"Datos de prueba: {len(X_test)}")

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )
    datagen.fit(X_train)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath="models/ocr_model_best.weights.h5", monitor='val_accuracy',
        save_best_only=True, verbose=1, save_weights_only=True
    )

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=64),
        steps_per_epoch=len(X_train) // 64,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=[reduce_lr, early_stop, checkpoint]
    )

    model.save(MODEL_WEIGHTS)

    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    print("Modelo entrenado y guardado correctamente.")

if __name__ == "__main__":
    train_model()