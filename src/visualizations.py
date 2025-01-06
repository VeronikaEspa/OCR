# visualizations.py

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from tensorflow.keras.models import load_model
from config import DATA_PATH, MODEL_WEIGHTS, HISTORY_PATH
from preprocess import preprocess_data

def plot_training_history(history_path=HISTORY_PATH):
    """
    Graficar la precisión y la pérdida durante el entrenamiento.
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
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
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
    """
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Reporte de Clasificación:\n")
    print(report)

def plot_precision_recall(model, X_test, y_test, class_names):
    """
    Plot Precision-Recall curves for each class.
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
    plt.legend(loc='best')
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
    if not os.path.exists("models/ocr_model.h5"):
        print(f"El archivo del modelo models/ocr_model.h5 no existe.")
        return
    if not os.path.exists(DATA_PATH):
        print(f"El archivo de datos {DATA_PATH} no existe.")
        return
    
    # Graficar historial de entrenamiento
    plot_training_history(HISTORY_PATH)
    
    # Preprocesar datos para obtener X_test y y_test
    X_train, X_test, y_train, y_test, num_classes = preprocess_data(DATA_PATH)
    
    # Cargar el modelo completo
    model = load_model("models/ocr_model.h5")
    
    # Definir nombres de clases (A-Z)
    class_names = [chr(i) for i in range(65, 65 + num_classes)]  # ['A', 'B', ..., 'Z']
    
    # Mostrar reporte de clasificación
    display_classification_report(model, X_test, y_test, class_names)
    
    # Generar y mostrar matriz de confusión
    plot_confusion_matrix(model, X_test, y_test, class_names)
    
    # Plot Precision-Recall curves
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
