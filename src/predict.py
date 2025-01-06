# predict.py
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image, preprocess_image_from_pil
import os
import matplotlib.pyplot as plt
from PIL import Image
from config import TESTING_PATH, MODEL_WEIGHTS
import cv2

def test_model_predictions():
    model = load_model("models/ocr_model_best.weights.h5")
    class_names = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    test_images = [f for f in os.listdir('test_images') if f.endswith('.png')]

    for image_path in test_images:
        img = preprocess_image(os.path.join('test_images', image_path))
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        print(f"Predicción: {class_names[predicted_class]}")

def splitting_letters_of_a_word():
    """
    Procesar palabras completas, dividiéndolas en letras y prediciendo.
    """
    subfolder = 'words'
    word_images = get_test_images(TESTING_PATH, subfolder)

    if not word_images:
        print("No hay imágenes de palabras para procesar.")
        return

    model = load_trained_model()
    if model is None:
        return

    class_names = [chr(i) for i in range(65, 91)]

    for word_image_path in word_images:
        img = Image.open(word_image_path).convert('L')
        img_array = np.array(img)

        # Segmentar letras
        _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])

        letters = []
        for box in bounding_boxes:
            x, y, w, h = box
            letter = img_array[y:y+h, x:x+w]
            letter = Image.fromarray(letter).resize((28, 28))
            processed_img = preprocess_image_from_pil(letter)  # Usa la nueva función

            prediction = model.predict(processed_img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_letter = class_names[predicted_class]
            letters.append(predicted_letter)

        print(f"Palabra procesada: {''.join(letters)}")

def load_trained_model():
    """
    Cargar el modelo entrenado completo.
    """
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"El archivo del modelo {MODEL_WEIGHTS} no existe.")
        return None

    try:
        model = load_model(MODEL_WEIGHTS)
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def get_test_images(testing_folder, subfolder):
    """
    Obtener imágenes de prueba.
    """
    folder_path = os.path.join(testing_folder, subfolder)
    if not os.path.exists(folder_path):
        print(f"La carpeta {folder_path} no existe.")
        return []

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
