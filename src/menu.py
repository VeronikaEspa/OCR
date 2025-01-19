from train_model import train_model
from predict import test_model_predictions, splitting_letters_of_a_word
from visualizations import evaluate_model
from performance import medir_tiempo

train_model = medir_tiempo(train_model)
test_model_predictions = medir_tiempo(test_model_predictions)
splitting_letters_of_a_word = medir_tiempo(splitting_letters_of_a_word)
evaluate_model = medir_tiempo(evaluate_model)

def display_menu():
    """
    Mostrar un menú interactivo para el usuario.
    """
    print("Bienvenido al OCR Casero")
    while True:
        print("\nMenú:")
        print("1. Entrenar modelo y guardar resultado")
        print("2. Predecir letra sola")
        print("3. Predecir palabra")
        print("4. Evaluar modelo")
        print("5. Salir")
        option = input("Seleccione una opción: ")

        if option == "1":
            train_model()
        elif option == "2":
            test_model_predictions()
        elif option == "3":
            splitting_letters_of_a_word()
        elif option == "4":
            evaluate_model()
        elif option == "5":
            print("\u00a1Hasta luego!")
            break
        else:
            print("Opción no válida. Intente de nuevo.")
