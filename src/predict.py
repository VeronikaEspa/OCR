# predict.py
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image, preprocess_image_from_pil
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from config import TESTING_PATH, MODEL_WEIGHTS
import cv2


def test_model_predictions():
    model = load_model(MODEL_WEIGHTS)
    class_names = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
    letters_path = os.path.join(TESTING_PATH, "letters")
    
    if not os.path.exists(letters_path):
        print(f"La carpeta {letters_path} no existe.")
        return
    
    test_images = [f for f in os.listdir(letters_path) if f.endswith('.png')]
    
    if not test_images:
        print(f"No se encontraron imágenes en {letters_path}.")
        return

    for image_path in test_images:
        full_image_path = os.path.join(letters_path, image_path)
        
        with Image.open(full_image_path) as img:
            img = img.convert("L")
            
            img_inverted = ImageOps.invert(img)
            img_array = np.array(img_inverted)

            coords = np.column_stack(np.where(img_array > 0))
            if coords.size == 0:
                print(f"La imagen {image_path} no contiene letras detectables.")
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            cropped = img_inverted.crop((x_min, y_min, x_max + 1, y_max + 1))

            resized = cropped.resize((20, 20), Image.BICUBIC)

            final_image = Image.new("L", (28, 28), 0)
            final_image.paste(resized, (4, 4))

            plt.figure(figsize=(6, 6))
            plt.imshow(final_image, cmap='gray')
            plt.title(f"Procesada: {image_path}", fontsize=16)
            plt.axis('off')
            plt.show()

            img_to_predict = np.array(final_image).astype("float32") / 255.0
            img_to_predict = np.expand_dims(np.expand_dims(img_to_predict, axis=-1), axis=0)

            prediction = model.predict(img_to_predict)
            predicted_class = np.argmax(prediction)
            predicted_letter = class_names[predicted_class]

            print(f"Predicción para {image_path}: {predicted_letter}")


def splitting_letters_of_a_word():
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
            
            # Invertir colores de las letras
            letter = cv2.bitwise_not(letter)
            
            letter = Image.fromarray(letter).resize((28, 28))
            processed_img = preprocess_image_from_pil(letter)

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
