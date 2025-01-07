# visualizations.py

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from tensorflow.keras.models import load_model
from config import DATA_PATH, DATA_ZIP_PATH, MODEL_WEIGHTS, HISTORY_PATH
from train_model import preprocess_and_split_data, load_npz_data, load_zip_data

def plot_training_history(history_path=HISTORY_PATH):
    """
    Graficar la precisión y la pérdida durante el entrenamiento.
    
    Args:
        history_path (str): Ruta al archivo del historial de entrenamiento.
    """
    if not os.path.exists(history_path):
        print(f"El archivo de historial de entrenamiento {history_path} no existe.")
        return
    
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    acc = history.get('accuracy', [])
    val_acc = history.get('val_accuracy', [])
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 6))
    
    # Gráfica de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precisión de Entrenamiento', marker='o')
    plt.plot(epochs_range, val_acc, label='Precisión de Validación', marker='o')
    plt.legend(loc='lower right')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.grid(True)
    
    # Gráfica de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento', marker='o')
    plt.plot(epochs_range, val_loss, label='Pérdida de Validación', marker='o')
    plt.legend(loc='upper right')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, class_names):
    """
    Generar y mostrar la matriz de confusión.
    
    Args:
        model (tensorflow.keras.Model): Modelo entrenado.
        X_test (np.ndarray): Datos de prueba.
        y_test (np.ndarray): Etiquetas de prueba (one-hot).
        class_names (list): Lista de nombres de clases.
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    plt.title('Matriz de Confusión')
    plt.show()

def display_classification_report(model, X_test, y_test, class_names):
    """
    Mostrar el reporte de clasificación con precisión, recall y F1-score.
    
    Args:
        model (tensorflow.keras.Model): Modelo entrenado.
        X_test (np.ndarray): Datos de prueba.
        y_test (np.ndarray): Etiquetas de prueba (one-hot).
        class_names (list): Lista de nombres de clases.
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Reporte de Clasificación:\n")
    print(report)

def plot_precision_recall(model, X_test, y_test, class_names):
    """
    Graficar las curvas Precision-Recall para cada clase.
    
    Args:
        model (tensorflow.keras.Model): Modelo entrenado.
        X_test (np.ndarray): Datos de prueba.
        y_test (np.ndarray): Etiquetas de prueba (one-hot).
        class_names (list): Lista de nombres de clases.
    """
    y_scores = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_scores[:, i])
        average_precision = average_precision_score(y_test[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {average_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas de Precisión-Recall por Clase')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    plt.show()

def evaluate_model():
    """
    Evaluar el modelo incluyendo todas las visualizaciones y métricas adicionales.
    """
    # Verificar existencia de archivos
    if not os.path.exists(HISTORY_PATH):
        print(f"El archivo de historial de entrenamiento {HISTORY_PATH} no existe.")
        return
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"El archivo del modelo {MODEL_WEIGHTS} no existe.")
        return
    if not os.path.exists(DATA_PATH):
        print(f"El archivo de datos {DATA_PATH} no existe.")
        return

    # Graficar historial de entrenamiento
    plot_training_history(HISTORY_PATH)

    # Cargar y preprocesar los datos
    npz_X, npz_y = load_npz_data(DATA_PATH, fraction=0.5)
    zip_X, zip_y, label_to_index = load_zip_data(DATA_ZIP_PATH, fraction=0.5, target_size=(28, 28))

    if npz_X.size == 0 or zip_X.size == 0:
        print("No se pudieron cargar datos de uno o ambos archivos.")
        return

    X_train, X_test, y_train, y_test, num_classes = preprocess_and_split_data(npz_X, npz_y, zip_X, zip_y)

    # Convertir etiquetas a one-hot encoding si no lo están
    y_train = np.eye(num_classes)[y_train] if y_train.ndim == 1 else y_train
    y_test = np.eye(num_classes)[y_test] if y_test.ndim == 1 else y_test

    # Cargar el modelo completo
    model = load_model(MODEL_WEIGHTS)

    # Obtener los nombres de las clases desde el mapeo de etiquetas
    class_names = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]

    # Mostrar reporte de clasificación
    display_classification_report(model, X_test, y_test, class_names)

    # Generar y mostrar matriz de confusión
    plot_confusion_matrix(model, X_test, y_test, class_names)

    # Graficar curvas Precision-Recall
    plot_precision_recall(model, X_test, y_test, class_names)

    # Calcular y mostrar métricas de fiabilidad
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    overall_accuracy = np.mean(y_pred == y_true) * 100
    print(f"Exactitud Global del Modelo: {overall_accuracy:.2f}%")

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    precision_avg = np.mean([report[cls]['precision'] for cls in class_names])
    recall_avg = np.mean([report[cls]['recall'] for cls in class_names])
    f1_avg = np.mean([report[cls]['f1-score'] for cls in class_names])

    print(f"Precisión Promedio: {precision_avg * 100:.2f}%")
    print(f"Recall Promedio: {recall_avg * 100:.2f}%")
    print(f"F1-Score Promedio: {f1_avg * 100:.2f}%")

    # Recomendación basada en métricas
    if overall_accuracy < 90:
        print("El modelo necesita mejoras. Considera recolectar más datos, ajustar hiperparámetros o mejorar la arquitectura del modelo.")
    else:
        print("El modelo tiene un rendimiento satisfactorio.")
