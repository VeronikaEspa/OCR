source venv/bin/activate

pip install -r requirements.txt

Python 3.11.9

## Lista de Tareas

### Características por Implementar
#### Tipografias
##### Mayusculas
- [X] Reconocimiento de letras
- [X] Reconocimiento de una palabra
<!-- - [ ] Reconocimiento de palabras fotografiada -->

##### Minusculas
- [ ] Reconocimiento de letras
- [ ] Reconocimiento de una palabra
- [ ] Reconocimiento de palabras fotografiada

##### Mixto
- [ ] Reconocimiento de letras
- [ ] Reconocimiento de una palabra
- [ ] Reconocimiento de palabras fotografiada

#### Escrito a mano
##### Mayusculas
- [ ] Reconocimiento de letras
- [ ] Reconocimiento de una palabra
- [ ] Reconocimiento de palabras fotografiada

##### Minusculas
- [ ] Reconocimiento de letras
- [ ] Reconocimiento de una palabra
- [ ] Reconocimiento de palabras fotografiada

##### Mixto
- [ ] Reconocimiento de letras
- [ ] Reconocimiento de una palabra
- [ ] Reconocimiento de palabras fotografiada




Descripción del Diagrama
Módulos Principales:

main.py: Punto de entrada del proyecto que llama a display_menu desde menu.py.
menu.py: Presenta un menú interactivo y llama a las funciones correspondientes según la opción seleccionada.
Módulos Funcionales:

train_model.py: Maneja el entrenamiento del modelo OCR, incluyendo la compilación, el entrenamiento y el guardado de pesos e historial.
predict.py: Gestiona las predicciones del modelo tanto para letras individuales como para palabras completas, incluyendo la carga del modelo entrenado y el preprocesamiento de imágenes.
visualizations.py: Encargado de generar visualizaciones como el historial de entrenamiento, matrices de confusión y curvas de precisión-recall, además de evaluar el modelo.
preprocess.py: Contiene funciones para preprocesar imágenes y datos necesarios para el entrenamiento y las predicciones.
model.py: Define la arquitectura del modelo CNN utilizado para la clasificación de caracteres.
config.py: Almacena configuraciones y constantes utilizadas en todo el proyecto, como rutas de datos, pesos del modelo y parámetros de entrenamiento





@startuml
!define RECTANGLE class

' Definir los módulos como componentes
package "main" {
    RECTANGLE Main {
        + main()
    }
}

package "menu" {
    RECTANGLE Menu {
        + display_menu()
    }
}

package "train_model" {
    RECTANGLE TrainModel {
        + train_ocr_model()
        + train_and_save_model()
        + compile_model()
    }
}

package "predict" {
    RECTANGLE Predict {
        + test_model_predictions()
        + splitting_letters_of_a_word()
        + load_trained_model()
        + get_test_images()
    }
}

package "visualizations" {
    RECTANGLE Visualizations {
        + plot_training_history()
        + evaluate_model()
        + plot_confusion_matrix()
        + display_classification_report()
        + plot_precision_recall()
    }
}

package "preprocess" {
    RECTANGLE Preprocess {
        + preprocess_image()
        + preprocess_data()
    }
}

package "model" {
    RECTANGLE Model {
        + build_model()
    }
}

package "config" {
    RECTANGLE Config {
        + IMAGE_SIZE
        + DATA_PATH
        + MODEL_WEIGHTS
        + TESTING_PATH
        + HISTORY_PATH
        + EPOCHS
    }
}

' Relaciones entre módulos
Main --> Menu : llama a
Menu --> TrainModel : llama a
Menu --> Predict : llama a
Menu --> Visualizations : llama a

TrainModel --> Preprocess : usa
TrainModel --> Model : usa
TrainModel --> Config : usa

Predict --> Preprocess : usa
Predict --> Model : usa
Predict --> Config : usa

Visualizations --> Preprocess : usa
Visualizations --> Model : usa
Visualizations --> Config : usa

Preprocess --> Config : usa
Model --> Config : usa

' Detalles de dependencias adicionales
TrainModel --> "tensorflow.keras" : importa
Predict --> "tensorflow.keras" : importa
Visualizations --> "matplotlib.pyplot" : importa
Visualizations --> "seaborn" : importa
Visualizations --> "sklearn.metrics" : importa

@enduml
