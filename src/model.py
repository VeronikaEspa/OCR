# model.py
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    """
    Construir el modelo CNN para OCR con regularización.
    """
    model = models.Sequential()
    
    # Primera capa convolucional
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Segunda capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Tercera capa convolucional
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Capa de aplanamiento
    model.add(layers.Flatten())
    
    # Capa densa completamente conectada
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Capa de salida
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model