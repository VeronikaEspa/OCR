import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from tensorflow.keras.models import load_model
from datetime import datetime
from train_model import preprocess_and_split, load_and_merge_datasets
from config import MODEL_WEIGHTS, HISTORY_PATH, RESULT_GRAFICAS


os.makedirs(RESULT_GRAFICAS, exist_ok=True)

def save_plot(fig, title):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULT_GRAFICAS, f"{title}_{timestamp}.png")
    fig.savefig(filename)
    plt.show()
    print(f"Gráfica guardada en: {filename}")

def plot_training_history(history_path=HISTORY_PATH):
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(epochs_range, acc, label='Precisión de Entrenamiento', marker='o')
    axes[0].plot(epochs_range, val_acc, label='Precisión de Validación', marker='o')
    axes[0].legend(loc='lower right')
    axes[0].set_title('Precisión durante el Entrenamiento')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Precisión')
    axes[0].grid(True)

    axes[1].plot(epochs_range, loss, label='Pérdida de Entrenamiento', marker='o')
    axes[1].plot(epochs_range, val_loss, label='Pérdida de Validación', marker='o')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Pérdida durante el Entrenamiento')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Pérdida')
    axes[1].grid(True)

    plt.tight_layout()
    save_plot(fig, "training_history")
    plt.close(fig)

def plot_confusion_matrix(model, X_test, y_test, class_names):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    plt.title('Matriz de Confusión')
    save_plot(fig, "confusion_matrix")
    plt.show()

def display_classification_report(model, X_test, y_test, class_names):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Reporte de Clasificación:\n")
    print(report)

def plot_precision_recall(model, X_test, y_test, class_names):
    y_scores = model.predict(X_test)

    fig = plt.figure(figsize=(14, 10))

    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test[:, i], y_scores[:, i])
        average_precision = average_precision_score(y_test[:, i], y_scores[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas de Precisión-Recall por Clase')
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    save_plot(fig, "precision_recall")
    plt.show()

def evaluate_model():
    if not os.path.exists(HISTORY_PATH):
        print(f"El archivo de historial de entrenamiento {HISTORY_PATH} no existe.")
        return
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"El archivo del modelo {MODEL_WEIGHTS} no existe.")
        return

    plot_training_history(HISTORY_PATH)

    images, labels = load_and_merge_datasets()
    X_train, X_test, y_train, y_test, num_classes = preprocess_and_split(images, labels)

    model = load_model(MODEL_WEIGHTS)

    class_names = sorted(set(labels))

    display_classification_report(model, X_test, y_test, class_names)
    plot_confusion_matrix(model, X_test, y_test, class_names)
    plot_precision_recall(model, X_test, y_test, class_names)

    train_accuracy = model.evaluate(X_train, y_train, verbose=0)[1] * 100
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1] * 100

    print(f"\nExactitud en datos de entrenamiento: {train_accuracy:.2f}%")
    print(f"Exactitud en datos de prueba: {test_accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_model()