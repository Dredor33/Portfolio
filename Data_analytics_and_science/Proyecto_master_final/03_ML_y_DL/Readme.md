# 03_ML_y_DL

Esta carpeta contiene los archivos relacionados con el **modelo de Deep Learning final** del proyecto, así como datos y resultados de prueba.  

## 📂 Contenido

- `Proyecto Júpiter VA DADS0225 Deep Learning.py` → Script principal con el modelo definitivo y comentarios explicativos. Explica el entrenamiento, fine-tuning y la arquitectura utilizada.  
- `modelo_final.keras` → Red neuronal final usada para la predicción.  
- `label_encoder.pkl` → Información de codificación de etiquetas utilizada durante el entrenamiento.  
- `productos ML DL.csv` → Extracto de datos del macrodataframe, preparado para entrenar y testear el modelo.  
- `resultados_test.csv` → Resultados de las predicciones realizadas con el modelo final.  
- `Modelo descartado/` y `Proyecto DCA descartado/` → Intentos de modelo anteriores que no dieron los resultados esperados.

## ⚡ Resumen del modelo

- Se utilizó como base un modelo preentrenado de Google, al que se le añadieron capas adicionales y una última capa softmax para determinar el porcentaje de confianza de las predicciones.  
- Se implementó fine-tuning sobre capas seleccionadas y se definió un umbral óptimo para maximizar la cantidad de imágenes procesadas con el menor número de errores.  
- Las predicciones que no cumplen con el umbral podrían enviarse a control humano o a un sistema de revisión adicional.

> Todo el proceso está explicado en el archivo `.py` y preparado para ser reproducido directamente.  
> ⚠️ Para poder ejecutar completamente el modelo haría falta tener acceso a las imágenes del proyecto (~8GB), por lo que no se incluyen en el repositorio.
