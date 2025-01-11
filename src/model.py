# model.py
from tensorflow.keras import layers, models, regularizers
from performance import medir_tiempo

def build_model(input_shape, num_classes):
    """
    Construir el modelo CNN para OCR con regularización mejorada.
    """
    model = models.Sequential()
    
    # Capa de entrada
    model.add(layers.Input(shape=input_shape))
    
    # Primera capa convolucional
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Segunda capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Tercera capa convolucional
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Capa de aplanamiento
    model.add(layers.Flatten())
    
    # Capa densa completamente conectada
    model.add(layers.Dense(512, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Segunda capa densa
    model.add(layers.Dense(256, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Capa de salida con 52 clases
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

build_model = medir_tiempo(build_model) 