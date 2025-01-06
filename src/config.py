# config.py

IMAGE_SIZE = (28, 28)

DATA_ZIP_PATH = "./data/training/english_alphabets.zip"
DATA_PATH = "./data/training/character_fonts (with handwritten data).npz"
TESTING_PATH = "./data/testing"

MODEL_WEIGHTS= "models/ocr_model.weights.h5"
MODEL_WEIGHTS_BEST= "models/ocr_model_best.weights.h5"
HISTORY_PATH = './models/training_history.pkl'

EPOCHS = 15