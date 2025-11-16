# Proyecto DCA descartado

Esta carpeta contiene un **intento fallido de modelo** que se exploró antes de llegar al enfoque final del proyecto Jupiter VA.  
El objetivo era experimentar con distintos métodos de procesamiento de imágenes y modelos de machine learning para identificar frutas, pero finalmente no se obtuvieron resultados satisfactorios.

## 📂 Contenido

- `Jupiter_ML.ipynb` → Notebook con el análisis y experimentos preliminares de procesamiento de imágenes, PCA, SVC, KNN y RandomForest. Incluye comentarios explicativos de los pasos seguidos.  
- `color_vectors_corregidos.npz` → Vector de medias de color de las imágenes para filtros iniciales.  
- `bins_comunes.pkl` → Datos generados en la exploración de pixeles comunes entre imágenes.  
- `dataset_completo.csv` → Extracto de los datos de imágenes utilizadas para los experimentos.

## ⚡ Resumen del enfoque

- Se exploraron técnicas de filtrado de pixeles comunes y únicos para intentar aislar las frutas del fondo.  
- Se probaron métodos de reducción de dimensionalidad (PCA) y varios modelos de clasificación (SVC, KNN, RandomForest) para encontrar clusters y clasificar las frutas.  
- Los experimentos mostraron limitaciones importantes: variaciones en posición, distancia y orientación de las imágenes, así como problemas de tiempo de ejecución en Colab.  
- Por estas razones, este enfoque fue descartado y se desarrolló posteriormente el modelo definitivo en la carpeta `03_ML_y_DL`.

> Este README refleja el proceso exploratorio y el aprendizaje obtenido, aunque el modelo no se utilizó en el proyecto final.
