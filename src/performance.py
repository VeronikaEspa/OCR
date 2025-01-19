import time
import functools

def medir_tiempo(func):
    """
    Decorador para medir el tiempo de ejecución de una función.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        tiempo_ejecucion = fin - inicio
        print(f"\n[PERFORMANCE] La función '{func.__name__}' se ejecutó en {tiempo_ejecucion:.4f} segundos.\n")
        return resultado
    return wrapper