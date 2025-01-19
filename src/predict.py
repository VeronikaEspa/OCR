# # predict.py
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
import os
from PIL import Image, ImageOps, ImageDraw
from config import TESTING_PATH, MODEL_WEIGHTS, RESULT_PREDICT
import cv2
import re
import matplotlib.pyplot as plt
from train_model import load_and_merge_datasets

def test_model_predictions():
    """
    Realiza predicciones sobre imágenes de letras individuales, calcula la precisión global,
    imprime detalles individuales en la consola y visualiza todas las imágenes en una única ventana
    con etiquetas claras.
    """
    # Cargar el modelo
    try:
        model = load_model(MODEL_WEIGHTS)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Obtener clases dinámicamente desde los datos
    _, labels = load_and_merge_datasets()
    class_names = sorted(set(labels))

    letters_path = os.path.join(TESTING_PATH, "letters")

    if not os.path.exists(letters_path):
        print(f"La carpeta {letters_path} no existe.")
        return

    test_images = [f for f in os.listdir(letters_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not test_images:
        print(f"No se encontraron imágenes en {letters_path}.")
        return

    total_images = 0
    correct_predictions = 0
    images_to_display = []
    actual_labels = []
    predicted_labels = []
    filenames = []

    for image_path in test_images:
        full_image_path = os.path.join(letters_path, image_path)

        try:
            img_to_predict, processed_img = preprocess_image(full_image_path)

            image_name_with_ext = os.path.basename(full_image_path)
            image_name, _ = os.path.splitext(image_name_with_ext)
            output_dir = os.path.join(RESULT_PREDICT, image_name)
            os.makedirs(output_dir, exist_ok=True)

            processed_img_pil = Image.fromarray((processed_img * 255).astype(np.uint8))  # Escalar valores a 0-255
            processed_img_path = os.path.join(output_dir, f"processed_{image_name_with_ext}")
            processed_img_pil.save(processed_img_path)
            print(f"Imagen preprocesada guardada en: {processed_img_path}")

            if len(img_to_predict.shape) == 2:
                img_to_predict = np.expand_dims(np.expand_dims(img_to_predict, axis=-1), axis=0)
            elif len(img_to_predict.shape) == 3:
                img_to_predict = np.expand_dims(img_to_predict, axis=0)

        except Exception as e:
            print(f"Error al preprocesar la imagen {image_path}: {e}")
            continue

        prediction = model.predict(img_to_predict)
        predicted_class = np.argmax(prediction)
        predicted_letter = class_names[predicted_class] if predicted_class < len(class_names) else '?'
        print(f"Imagen: {image_path}, Predicción: {predicted_letter}")

        actual_letter = re.sub(r'\(\d+\)$', '', image_name)

        total_images += 1

        if actual_letter.lower() == predicted_letter.lower():
            correct_predictions += 1
            is_correct = True
        else:
            is_correct = False

        print(f"Archivo: {image_name_with_ext} | Real: {actual_letter} | Predicción: {predicted_letter} | Correcto: {is_correct}")

        images_to_display.append(img_to_predict[0, :, :, 0])
        actual_labels.append(actual_letter)
        predicted_labels.append(predicted_letter)
        filenames.append(image_name_with_ext)

    if total_images == 0:
        print("No se procesaron imágenes para evaluar.")
    else:
        accuracy = (correct_predictions / total_images) * 100
        print(f"\nTotal de imágenes procesadas: {total_images}")
        print(f"Predicciones correctas: {correct_predictions}")
        print(f"Exactitud Global: {accuracy:.2f}%")

    if images_to_display:
        num_images = len(images_to_display)
        max_display = 20
        if num_images > max_display:
            print(f"Mostrando solo las primeras {max_display} imágenes para la visualización.")
            images_to_display = images_to_display[:max_display]
            actual_labels = actual_labels[:max_display]
            predicted_labels = predicted_labels[:max_display]
            filenames = filenames[:max_display]
            num_images = max_display

        cols = 5
        rows = num_images // cols + int(num_images % cols > 0)

        plt.figure(figsize=(15, 3 * rows))
        plt.subplots_adjust(hspace=0.6, wspace=0.4)

        for idx, (img, actual, pred, fname) in enumerate(zip(images_to_display, actual_labels, predicted_labels, filenames)):
            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Real: {actual}\nPred: {pred}", fontsize=10, pad=15)

        plt.tight_layout()
        plt.show()


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

import re

def splitting_letters_of_a_word():
    subfolder = 'words'
    word_images = get_test_images(TESTING_PATH, subfolder)

    if not word_images:
        print("No hay imágenes de palabras para procesar.")
        return

    model = load_trained_model()
    if model is None:
        return

    _, labels = load_and_merge_datasets()
    class_names = sorted(set(labels))

    total_accuracy = 0.0
    total_words = 0

    for word_image_path in word_images:
        try:
            img = Image.open(word_image_path).convert('L')
            inverted_img = ImageOps.invert(img)
            img_array = np.array(inverted_img)
        except Exception as e:
            print(f"Error al abrir la imagen {word_image_path}: {e}")
            continue

        image_name_with_ext = os.path.basename(word_image_path)
        image_name, _ = os.path.splitext(image_name_with_ext)

        # Eliminar todo lo que esté entre paréntesis y reemplazar espacios por vacío
        actual_word = re.sub(r'\(.*?\)', '', image_name).replace(' ', '').strip()

        output_dir = os.path.join(RESULT_PREDICT, actual_word)
        os.makedirs(output_dir, exist_ok=True)

        _, binary = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_with_boxes = inverted_img.convert('RGB')
        draw = ImageDraw.Draw(img_with_boxes)

        bounding_boxes = sorted(
            [cv2.boundingRect(c) for c in contours],
            key=lambda b: b[0]
        )

        letters = []
        for idx, box in enumerate(bounding_boxes):
            x, y, w, h = box

            if w * h < 5:
                continue

            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)

            letter = img_array[y:y + h, x:x + w]

            letter_image = Image.fromarray(letter)

            inverted_letter_image = ImageOps.invert(letter_image)
            letter_path = os.path.join(output_dir, f'letter_{idx + 1}.png')

            try:
                img_to_predict, processed_img = preprocess_image(inverted_letter_image)
                Image.fromarray((processed_img * 255).astype(np.uint8)).save(letter_path)

                prediction = model.predict(img_to_predict)
                predicted_class = np.argmax(prediction, axis=1)[0]

                if predicted_class < len(class_names):
                    predicted_label = class_names[predicted_class]
                else:
                    predicted_label = '?'

                letters.append(predicted_label)
            except Exception as e:
                print(f"Error procesando la letra: {e}")
                letters.append('?')

        predicted_word = ''.join(letters)

        correct_letters = 0
        total_letters = min(len(actual_word), len(predicted_word))

        for i in range(total_letters):
            if actual_word[i].lower() == predicted_word[i].lower():
                correct_letters += 1

        if len(actual_word) == 0:
            accuracy = 0.0
        else:
            accuracy = (correct_letters / len(actual_word)) * 100

        total_accuracy += accuracy
        total_words += 1

        print(f"Palabra procesada: {predicted_word} del archivo {image_name_with_ext} - Exactitud: {accuracy:.2f}%")

    if total_words > 0:
        overall_accuracy = total_accuracy / total_words
        print(f"\nPorcentaje de acierto acumulado: {overall_accuracy:.2f}%")
    else:
        print("No se procesaron palabras para calcular el porcentaje de acierto.")


def get_test_images(testing_folder, subfolder):
    """
    Obtener imágenes de prueba.
    """
    folder_path = os.path.join(testing_folder, subfolder)
    if not os.path.exists(folder_path):
        print(f"La carpeta {folder_path} no existe.")
        return []

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]