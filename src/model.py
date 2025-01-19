from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from performance import medir_tiempo
from config import IMAGE_SIZE, MODEL_WEIGHTS, MODEL_WEIGHTS_BEST, EPOCHS

@medir_tiempo
def build_model(input_shape, num_classes):
    """
    Construir el modelo CNN para OCR con regularización mejorada y arquitectura más profunda.
    
    Args:
        input_shape (tuple): Forma de la entrada de la imagen (alto, ancho, canales).
        num_classes (int): Número total de clases para la clasificación.
    
    Returns:
        tensorflow.keras.Model: Modelo CNN construido.
    """
    if input_shape[0] < 32 or input_shape[1] < 32:
        raise ValueError(f"Input size must be at least 32x32; Received: input_shape={input_shape}")

    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model

def compile_model(model):
    """
    Compilar el modelo con un optimizador, pérdida y métricas ajustadas.
    
    Args:
        model (tensorflow.keras.Model): Modelo a compilar.
    """
    optimizer = Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

def get_callbacks():
    """
    Obtener callbacks para mejorar el entrenamiento del modelo.

    Returns:
        list: Lista de callbacks configurados.
    """
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    return [lr_scheduler, early_stopping]

def data_augmentation():
    """
    Crear un generador de datos con técnicas de aumento de datos para OCR.
    
    Returns:
        tensorflow.keras.Sequential: Pipeline de data augmentation.
    """
    return tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.GaussianNoise(0.01),
    ])

if __name__ == "__main__":
    input_shape = (*IMAGE_SIZE, 1)
    num_classes = 62

    try:
        model = build_model(input_shape, num_classes)

        compile_model(model)

        data_gen = data_augmentation()

        callbacks = get_callbacks()
        model.save_weights(MODEL_WEIGHTS)

    except ValueError as e:
        print(f"Error: {e}")