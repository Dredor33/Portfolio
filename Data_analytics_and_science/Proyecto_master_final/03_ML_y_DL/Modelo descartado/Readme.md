 Modelo descartado

Esta carpeta contiene un **modelo de red neuronal experimental** que se probó antes de decidir el modelo final del proyecto Jupiter VA.  
El objetivo era explorar distintas arquitecturas de red para la clasificación de frutas a partir de imágenes, pero finalmente se eligió otro enfoque más efectivo.

## 📂 Contenido

- `Jupiter_nn.ipynb` → Notebook con la experimentación de la red neuronal, incluyendo capas convolucionales, capas 2D, flatten, dense y softmax. Se realizaron 6 epochs y se obtuvieron resultados decentes.  
- `dataset_completo.csv` → Información de las imágenes y etiquetas utilizadas para entrenar y evaluar la red neuronal.

## ⚡ Resumen del enfoque

- Se probó una arquitectura convolucional con varias capas, seguida de flatten y dense, terminando en softmax para clasificación.  
- Los resultados fueron prometedores, pero no alcanzaron la precisión y estabilidad del modelo definitivo desarrollado en la carpeta `03_ML_y_DL`.  
- Este modelo quedó como alternativa plausible y referencia para futuras exploraciones, pero no se utilizó en el proyecto final.

> Este README refleja la experimentación y aprendizaje obtenido, aunque este modelo fue descartado en favor del modelo final.
