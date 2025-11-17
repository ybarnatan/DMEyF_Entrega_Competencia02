# 📘 DMEyF 2025 — Maestría en Explotación de Datos y Descubrimiento del Conocimiento (FCEN – UBA)
## Competencia 02 — Pipeline Completo de Modelado de Churn en Clientes Premium

Este proyecto corresponde a la **Competencia 02** de la materia **DMEyF** y consiste en desarrollar un **pipeline integral de modelado predictivo** para estimar la probabilidad de *churn* (fuga) en clientes del segmento *premium* de un banco.

El modelo principal utilizado es **zLightGBM**, una variante personalizada de LightGBM que incorpora la lógica de *canaritos* para la detección de sobreajuste y la validación de integridad del pipeline.

---

## 🚀 Objetivo del Proyecto

Construir un **pipeline reproducible de punta a punta**, que incluye:

- Limpieza y enriquecimiento de datos  
- Feature engineering  
- Entrenamiento del modelo zLightGBM  
- Evaluación del modelo  
- Generación de predicciones finales

---

## 📂 Estructura fundamental del proyecto

```
├── data/                # Datos crudos y procesados
├── models/              # Modelos entrenados y artefactos
├── src/                 # Módulos del pipeline (EDA, features, entrenamiento, etc.)
├── outputs/             # Predicciones generadas para submit
├── main.py              # Script principal del pipeline completo
├── README.md            # Documentación del proyecto
└── requirements.txt     # Dependencias del entorno
```

---

## 🧠 Modelo Utilizado: zLightGBM

zLightGBM es una adaptación de LightGBM que incorpora:

- Canaritos para control de generalización  
- Ajustes específicos para alta dimensionalidad  como el subsampleo.


---

## ▶️ Cómo ejecutar el pipeline completo

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar el pipeline completo:

```bash
python main.py
```

Esto generará el modelo entrenado y el archivo de predicciones finales listo para submit.

---

## 🧪 Requisitos y dependencias

Principales librerías utilizadas:

- pandas  
- numpy  
- lightgbm / zlightgbm  
- matplotlib / seaborn  
- pyyaml  


---

## 📊 Resultados

El pipeline produce:

- Predicciones de churn para el conjunto de evaluación  
- Logs del proceso  
- Modelo entrenado  
- Métricas internas del desempeño  
- Archivo final para submit  
