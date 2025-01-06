# inspect_labels.py
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_PATH, DATA_ZIP_PATH
import random
import zipfile
from io import BytesIO
from PIL import Image
import os

def inspect_npz_labels(data_path):
    """
    Cargar los datos desde un archivo .npz, mostrar la distribución de etiquetas
    y visualizar una imagen aleatoria por clase.
    
    Args:
        data_path (str): Ruta al archivo .npz que contiene 'images' y 'labels'.
    """
    print("Analizando archivo .npz...")
    # Cargar los datos desde el archivo .npz
    try:
        data = np.load(data_path)
        X = data['images']
        y = data['labels']
    except FileNotFoundError:
        print(f"El archivo {data_path} no se encontró.")
        return
    except KeyError:
        print(f"El archivo {data_path} no contiene las claves 'images' y 'labels'.")
        return
    
    # Verificar las formas de los datos
    print(f"Forma de X (imágenes): {X.shape}")
    print(f"Forma de y (etiquetas): {y.shape}\n")
    
    # Obtener la distribución de etiquetas
    unique, counts = np.unique(y, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    
    # Asumiendo que las etiquetas son índices que corresponden a letras (0->'A', 1->'B', ..., 25->'Z')
    # Ajusta esto si tus etiquetas corresponden a otra cosa
    class_names = [chr(i) for i in range(65, 65 + len(unique))]  # ['A', 'B', ..., 'Z']
    label_to_class = {label: class_name for label, class_name in zip(unique, class_names)}
    
    print("Distribución de etiquetas en .npz:")
    for label, count in label_distribution.items():
        class_name = label_to_class.get(label, 'Desconocida')
        print(f"Etiqueta {label} ({class_name}): {count} imágenes")
    
    # Visualizar una imagen aleatoria por clase
    print("\nMostrando un ejemplo aleatorio de cada clase en .npz:")
    
    # Configurar el tamaño de la figura según el número de clases
    num_classes = len(unique)
    cols = 6  # Número de columnas en la grilla
    rows = num_classes // cols + int(num_classes % cols > 0)
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for idx, label in enumerate(unique):
        # Encontrar los índices de todas las imágenes con la etiqueta actual
        indices = np.where(y == label)[0]
        
        if len(indices) == 0:
            print(f"No hay imágenes para la etiqueta {label}.")
            continue
        
        # Seleccionar un índice aleatorio
        example_idx = random.choice(indices)
        example_image = X[example_idx]
        
        # Manejar diferentes formas de imagen
        if example_image.ndim == 3 and example_image.shape[2] == 1:
            # Imagen con un canal
            example_image = example_image.squeeze(axis=2)
        elif example_image.ndim == 2:
            # Imagen en escala de grises
            pass
        else:
            print(f"Imagen {example_idx} tiene una forma inesperada: {example_image.shape}.")
            continue
        
        # Verificar si la imagen está normalizada (asumiendo que los valores están entre 0 y 1)
        if example_image.max() <= 1.0:
            example_image_display = (example_image * 255).astype(np.uint8)
        else:
            example_image_display = example_image.astype(np.uint8)
        
        # Obtener el nombre de la clase
        class_name = label_to_class.get(label, 'Desconocida')
        
        # Crear un subplot para la imagen
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(example_image_display, cmap='gray')
        plt.title(f"Etiqueta {label}: {class_name}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def inspect_zip_labels(zip_path):
    """
    Cargar los datos desde un archivo .zip, mostrar la distribución de etiquetas
    y visualizar una imagen aleatoria por clase.
    
    Args:
        zip_path (str): Ruta al archivo .zip que contiene la carpeta 'dataset1' con subcarpetas de etiquetas.
    """
    print("\nAnalizando archivo .zip...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Listar todos los archivos en el zip
            all_files = zip_ref.namelist()
            
            # Filtrar los archivos que están dentro de 'dataset1/' y no son directorios
            dataset_files = [f for f in all_files if f.startswith('dataset1/') and not f.endswith('/')]
            
            if not dataset_files:
                print("No se encontraron archivos dentro de 'dataset1/' en el zip.")
                return
            
            # Extraer las etiquetas de los nombres de las carpetas
            labels = set()
            label_to_files = {}
            for file in dataset_files:
                parts = file.split('/')
                if len(parts) < 3:
                    continue  # Esperamos 'dataset1/etiqueta/imagen'
                label = parts[1]
                labels.add(label)
                label_to_files.setdefault(label, []).append(file)
            
            labels = sorted(labels)
            label_indices = {label: idx for idx, label in enumerate(labels)}
            label_distribution = {label: len(files) for label, files in label_to_files.items()}
            
            print("Distribución de etiquetas en .zip:")
            for label, count in label_distribution.items():
                print(f"Etiqueta '{label}': {count} imágenes")
            
            # Visualizar una imagen aleatoria por clase
            print("\nMostrando un ejemplo aleatorio de cada clase en .zip:")
            
            num_classes = len(labels)
            cols = 6  # Número de columnas en la grilla
            rows = num_classes // cols + int(num_classes % cols > 0)
            plt.figure(figsize=(cols * 3, rows * 3))
            
            for idx, label in enumerate(labels):
                files = label_to_files.get(label, [])
                if not files:
                    print(f"No hay imágenes para la etiqueta '{label}'.")
                    continue
                
                # Seleccionar un archivo aleatorio
                example_file = random.choice(files)
                
                # Leer la imagen desde el zip
                with zip_ref.open(example_file) as image_file:
                    image_data = image_file.read()
                    image = Image.open(BytesIO(image_data))
                    image = image.convert('L')  # Convertir a escala de grises si es necesario
                    image_np = np.array(image)
                
                # Verificar si la imagen está normalizada
                if image_np.max() <= 1.0:
                    image_display = (image_np * 255).astype(np.uint8)
                else:
                    image_display = image_np.astype(np.uint8)
                
                # Crear un subplot para la imagen
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(image_display, cmap='gray')
                plt.title(f"Etiqueta '{label}'")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    except FileNotFoundError:
        print(f"El archivo {zip_path} no se encontró.")
    except zipfile.BadZipFile:
        print(f"El archivo {zip_path} no es un archivo zip válido.")

if __name__ == "__main__":
    inspect_npz_labels(DATA_PATH)
    inspect_zip_labels(DATA_ZIP_PATH)
