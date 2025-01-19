# OCR

Este proyecto implementa un sistema de reconocimiento óptico de caracteres (OCR) utilizando TensorFlow y herramientas personalizadas para el procesamiento y evaluación de imágenes.

## Estructura del Proyecto

### `main.py`
Punto de entrada del programa. Invoca el menú interactivo para que el usuario realice diferentes tareas.

### `menu.py`
Contiene el menú principal que conecta al usuario con funcionalidades clave como el entrenamiento del modelo, predicciones y evaluaciones.

### `model.py`
Define la arquitectura del modelo de red neuronal convolucional (CNN), optimizado para OCR. Incluye funciones para construir, compilar y configurar callbacks.

### `performance.py`
Proporciona un decorador para medir y registrar el tiempo de ejecución de funciones críticas.

### `predict.py`
Incluye funciones para:
- Predecir caracteres individuales.
- Dividir palabras en letras y predecirlas.
- Gestionar el modelo entrenado y realizar pruebas sobre conjuntos de datos.

### `preprocess.py`
Contiene funciones para preprocesar imágenes, asegurando que estén listas para la predicción o entrenamiento. Incluye técnicas como recorte, redimensionamiento y normalización.

### `train_model.py`
Se encarga de:
- Cargar y combinar datasets (MNIST, EMNIST, ZIP, etc.).
- Preprocesar datos.
- Entrenar el modelo CNN.
- Guardar el modelo y su historial de entrenamiento.

### `visualizations.py`
Proporciona herramientas para evaluar el rendimiento del modelo mediante:
- Historial de entrenamiento.
- Matrices de confusión.
- Reportes de clasificación.
- Curvas de precisión-recall.

### `config.py`
Define constantes esenciales como rutas de datos, tamaño de imágenes y parámetros de entrenamiento.