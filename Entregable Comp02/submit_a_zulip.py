import pandas as pd
import os

'''

Tradicionalmente generabamos un archivo para subir a kaggle con formato id_cliente, prediccion (1 o 0).
Ahora lo piden en formato: lista directa de Id_cliente, sin headers ni nada.
Esta funcion accessoria hace eso
'''

def prediccion_zulip_comp02(path_entrada_csv, nombre_archivo_salida, path_salida=None):
    """
    Carga un archivo CSV, filtra las filas donde 'prediced' es igual a 1, 
    selecciona solo la columna 'numero_de_cliente', y guarda el resultado
    en un nuevo archivo CSV sin la fila de encabezado.

    Args:
        path_entrada_csv (str): La ruta completa del archivo CSV de entrada.
        nombre_archivo_salida (str): El nombre del archivo CSV de salida (e.g., 'clientes_filtrados.csv').
        path_salida (str, opcional): La carpeta donde se guardará el archivo. 
                                     Si es None, se usa la carpeta actual.
    
    Returns:
        str: La ruta completa del archivo de salida guardado.
    """
    try:
        # 1. Cargar el CSV
        df = pd.read_csv(path_entrada_csv)

        # 2. Filtrar por valor 'prediced' = 1
        df_filtrado = df[df['Predicted'] == 1]

        # 3. Quedarse solo con la columna 'numero_de_cliente'
        df_clientes = df_filtrado[['numero_de_cliente']]

        # 4. Construir la ruta de salida
        if path_salida:
            # Asegura que la carpeta de salida exista
            os.makedirs(path_salida, exist_ok=True)
            ruta_salida_completa = os.path.join(path_salida, nombre_archivo_salida)
        else:
            ruta_salida_completa = nombre_archivo_salida
        
        # 5. Guardar el CSV sin los headers (header=False)
        df_clientes.to_csv(ruta_salida_completa, index=False, header=False)

        return f"Archivo guardado con éxito en: {ruta_salida_completa}"

    except FileNotFoundError:
        return  "ERROR: El archivo de entrada no se encontró en la ruta especificada."
    except KeyError:
        return " ERROR: El CSV no contiene las columnas 'prediced' o 'numero_de_cliente'."
    except Exception as e:
        return f"Ocurrió un error inesperado: {e}"
    
#==
# --- Configuración de rutas ---

# Ruta de tu archivo CSV original (reemplaza con tu ruta real)
ruta_csv_entrada = 'C:/Users/ybbar/OneDrive/Desktop/Entregable Competencia 02 16Nov2025/outputs_experimentos_outputs_prediction_experimento_p_zlgbm_yb_LGBM_5_SEMILLAS_MES_TEST_202106_SEMILLA_ensamble_semillas_fase_testeoprediccion_test_proba.csv' 
# Nombre que quieres darle al archivo de salida
nombre_archivo_salida = 'Exp_p_zLGBM_16Nov2025_prediccionZulip.csv' 
# Carpeta de destino (None para guardar en la carpeta actual, o especifica una ruta)
carpeta_salida = 'C:/Users/ybbar/OneDrive/Desktop/Entregable Competencia 02 16Nov2025' 



# --- Llamada a la función ---
prediccion_zulip_comp02(
    path_entrada_csv=ruta_csv_entrada,
    nombre_archivo_salida=nombre_archivo_salida,
    path_salida=carpeta_salida
)
