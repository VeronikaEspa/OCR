# train_model.py
from config import MODEL_WEIGHTS, HISTORY_PATH, DATA_PATH, EPOCHS
from preprocess import preprocess_data
from model import build_model
import os
import pickle

def train_ocr_model():
    if not os.path.exists(DATA_PATH):
        print(f"El archivo {DATA_PATH} no existe.")
        return

    X_train, X_test, y_train, y_test, num_classes = preprocess_data(DATA_PATH)

    model = build_model(X_train.shape[1:], num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS)

    # Crear directorio para modelos si no existe
    os.makedirs(os.path.dirname(MODEL_WEIGHTS), exist_ok=True)
    model.save("models/ocr_model.h5")
    
    # Crear directorio para historial si no existe
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)

    print("Modelo entrenado y guardado correctamente.")
