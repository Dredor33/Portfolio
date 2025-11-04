# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 08:54:23 2025

@author: Propietario
"""

# ==================================================
# 1. IMPORTACIÓN LIBRERÍAS
# ==================================================
import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import unicodedata
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import shutil
from PIL import Image, ImageDraw, ImageFont


# Semillas para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Rutas
IMG_DIR = r"C:\\Users\\Propietario\\Desktop\\Máster Data Analytics & Data Science\\6.PROYECTO JÚPITER\\Visión artificial\\archive"
CSV_PATH = r"C:\\Users\\Propietario\\Desktop\\Máster Data Analytics & Data Science\\6.PROYECTO JÚPITER\\Visión artificial\\sql\\productos ML DL.csv"
MODEL_PATH = "modelo_final.keras"
ENCODER_PATH = "label_encoder.pkl"
OUTPUT_CSV = "resultados_test.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 5
EPOCHS_FINE = 10

# ==================================================
# 2. CARGA DE DATOS Y PREPROCESAMIENTO
# ==================================================

#Normalizamos nombres quitando cadenas a minúsculas, quita espacios/guiones/acentos
## para generar claves comparables entre nombres de las imágenes e IDs del CSV.

def normalize_name(name):
    name = str(name).strip().lower().replace(" ", "").replace("_", "")
    return unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")

# Mapeamos imágenes 
file_map = {}
for root, _, files in os.walk(IMG_DIR):
    for fn in files:
        name, ext = os.path.splitext(fn)
        if ext.lower() in ('.jpg', '.jpeg', '.png'):
            key = normalize_name(name)
            file_map[key] = os.path.join(root, fn)

# Cargamos CSV y fusionamos
df = pd.read_csv(CSV_PATH, dtype={'t_id': str})
df.columns = df.columns.str.strip().str.lower()
df['stem_raw'] = df['t_id'].apply(lambda x: str(x).split('.', 1)[0])
df['stem_norm'] = df['stem_raw'].apply(normalize_name)
df['filepath'] = df['stem_norm'].map(file_map)
df = df.dropna(subset=['filepath']).copy()
df['label'] = df['tipo'].astype(str)


# Codificación de etiquetas mediante label_encoder
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)

# ==================================================
# 3️. DIVISIÓN EN TRAIN, VALIDACIÓN Y TEST
# ==================================================
# Realizamos la división partiendo el dataset en 70% Train, 15% val y 15% test
# Stratify nos ayuda a mantener la proporción de clases en cada split
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['encoded_label'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['encoded_label'], random_state=SEED)

# ==================================================
# 4️. GENERADORES DE IMÁGENES
# ==================================================
# Usamos rescale para normalizar pixeles a [0,1]
# Usamos train_datagen para aplicar augmentations controladas (flip horizontal, cambios de brillo,
# ligero zoom) para aumentar variabilidad del training y reducir overfitting

train_datagen = ImageDataGenerator(rescale=1./255, brightness_range=[0.8, 1.2], horizontal_flip=True, zoom_range=0.1)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filepath', y_col='label', target_size=IMG_SIZE, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
val_generator = val_datagen.flow_from_dataframe(val_df, x_col='filepath', y_col='label', target_size=IMG_SIZE, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False)
test_generator = test_datagen.flow_from_dataframe(test_df, x_col='filepath', y_col='label', target_size=IMG_SIZE, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False)

# ==================================================
# 5️. CALCULAR PESOS DE CLASE
# ==================================================
# Para reducir el sesgo hacia clases con muchas imágenes usamos class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(df['encoded_label']), y=df['encoded_label'])
class_weights = dict(enumerate(class_weights))

# ==================================================
# 6️. CONSTRUCCIÓN DEL MODELO CON MOBILENETV3 (ETAPA 1: CONGELADO)
# ==================================================
#Usamos transfer learning: backbone preentrenado en ImageNet que extrae features
base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # congelado

x = base_model.output
x = Conv2D(16, (1, 1), activation='relu')(x)
x = MaxPooling2D()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==================================================
# 7️. CALLBACKS
# ==================================================
# Realizamos control automático para evitar sobreentrenamiento y ajustar Learning Rate
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# ==================================================
# 8️. ENTRENAMIENTO (FASE 1 - CAPAS CONGELADAS)
# ==================================================
history1 = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_FROZEN, class_weight=class_weights, callbacks=[checkpoint, lr_scheduler, early_stop])

# ==================================================
# 9️. FINE-TUNING (FASE 2 - CAPAS DESCONGELADAS)
# ==================================================
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_FINE, class_weight=class_weights, callbacks=[checkpoint, lr_scheduler, early_stop])

# ==================================================
# 10. EVALUACIÓN FINAL SOBRE TEST
# ==================================================
model = load_model(MODEL_PATH)
preds = model.predict(test_generator, verbose=1)
y_true = test_df['label'].values
y_pred_idx = np.argmax(preds, axis=1)
y_pred = label_encoder.inverse_transform(y_pred_idx)

# Creamos el reporte y la matriz de confusión para detectar qué clases se confunden entre sí.
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred, labels=label_encoder.classes_)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matriz de Confusión (Test)")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.tight_layout()
plt.show()

# Guardamos CSV de predicciones
results_df = pd.DataFrame({
    't_id': test_df['t_id'].values,   # añadimos el ID
    'filepath': test_df['filepath'].values,
    'real': y_true,
    'predicho': y_pred
})

results_df.to_csv(OUTPUT_CSV, index=False)

# ==================================================
# 11. GUARDAR EL LABEL_ENCODER
# ==================================================
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

# ==========================================
# 12. GRÁFICO DE MÉTRICAS (TRAIN, VAL, TEST)
# ==========================================

# Evaluamos en test
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

# Unimos históricos de train y val
full_history = history1.history.copy()
for key in history2.history:
    full_history[key] += history2.history[key]

# Agregamos test accuracy/loss como líneas horizontales
epochs = range(1, len(full_history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, full_history['accuracy'], label='Train Acc')
plt.plot(epochs, full_history['val_accuracy'], label='Val Acc')
plt.hlines(test_accuracy, 1, len(epochs), colors='red', linestyles='dashed', label='Test Acc')
plt.title('Accuracy')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, full_history['loss'], label='Train Loss')
plt.plot(epochs, full_history['val_loss'], label='Val Loss')
plt.hlines(test_loss, 1, len(epochs), colors='red', linestyles='dashed', label='Test Loss')
plt.title('Loss')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# 13. VISUALIZAR ERRORES EN EL TEST SET
# ==========================================
# Filtramos las predicciones incorrectas para su revisión manual
wrong_preds = results_df[results_df['real'] != results_df['predicho']]

#Calculamos tasa de error global y precisión general sobre el test
print(f"\n❌ Total de errores en test: {len(wrong_preds)} de {len(results_df)} muestras")

for i, row in wrong_preds.head(10).iterrows():
    img = plt.imread(row['filepath'])
    plt.imshow(img)
    plt.title(f"Real: {row['real']} | Predicho: {row['predicho']}")
    plt.axis('off')
    plt.show()   
    
# ==========================================
# 14. VISUALIZAR EN UNIDADES
# ==========================================
#Hacemos recuento de las imágenes mal clasificadas, calculamos el % de error y el % de precisión global
errores = (results_df['real'] != results_df['predicho']).sum()

total = len(results_df)

porcentaje_error = (errores / total) * 100
porcentaje_precision = 100 - porcentaje_error

# Mostramos resultados
print(f"📦 Total de imágenes evaluadas: {total}")
print(f"❌ Imágenes mal clasificadas: {errores} ({porcentaje_error:.2f}%)")
print(f"✅ Precisión general en test: {porcentaje_precision:.2f}%")


# Carpeta de salida para guardar las imágenes mal clasificadas
OUTPUT_DIR = "errores_clasificacion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fuente para escribir texto 
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for i, row in wrong_preds.iterrows():
    img = Image.open(row['filepath']).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Texto a escribir
    texto = f"ID: {row['t_id']} | Real: {row['real']} | Predicho: {row['predicho']}"

    # Medir ancho y alto del texto
    bbox = draw.textbbox((0, 0), texto, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Fondo para que el texto sea legible
    draw.rectangle([0, 0, text_w + 10, text_h + 10], fill=(0, 0, 0))
    draw.text((5, 5), texto, font=font, fill=(255, 255, 255))

    # Guardar imagen en carpeta de errores
    filename = os.path.basename(row['filepath'])
    img.save(os.path.join(OUTPUT_DIR, filename))

print(f"✅ Se guardaron {len(wrong_preds)} imágenes mal clasificadas en '{OUTPUT_DIR}'")


# ==========================================
# 15. VISUALIZAR IMAGENES CONFIANZA INFERIOR AL 90%
# ==========================================
# Obtenemos la confianza máxima por imagen
max_confidences = np.max(preds, axis=1)

# Número de imágenes con confianza < 90%
baja_confianza = np.sum(max_confidences < 0.90)
total_imagenes = len(max_confidences)
porcentaje_baja_confianza = (baja_confianza / total_imagenes) * 100

print(f"📊 Total de imágenes evaluadas: {total_imagenes}")
print(f"❗ Imágenes con confianza < 90%: {baja_confianza} ({porcentaje_baja_confianza:.2f}%)")

# Añadimos las confianzas al DataFrame de resultados
results_df['confianza'] = max_confidences

# Filtramos predicciones de baja confianza
bajas_df = results_df[results_df['confianza'] < 0.90]

# Mostramos ejemplos para su revisión
print("\n🔍 Ejemplos de predicciones con baja confianza:")
print(bajas_df[['filepath', 'real', 'predicho', 'confianza']].head(10))

# ==========================================
# 16. GRÁFICO DE DISTRIBUCIÓN DE CONFIANZAS
# ==========================================
plt.figure(figsize=(10, 6))
sns.histplot(results_df['confianza'], bins=20, kde=True, color="skyblue", edgecolor="black")
plt.axvline(0.9, color="red", linestyle="--", label="Umbral 90%")
plt.title("Distribución de las Confianzas de las Predicciones")
plt.xlabel("Confianza")
plt.ylabel("Número de Imágenes")
plt.legend()
plt.tight_layout()
plt.show()

#El gráfico nos muestra que el modelo es sólido y consistente, ya que la mayoría de predicciones tienen
#una confianza muy alta (cerca del 1.0).

# ==========================================
# 17. GRÁFICO DE BARRAS: % Alta vs Baja Confianza
# ==========================================
#Aanalizamos las imagenes que quedarían fuera con una confianza inferior al 90%.
labels = ["Confianza ≥ 90%", "Confianza < 90%"]
values = [total_imagenes - baja_confianza, baja_confianza]

plt.figure(figsize=(6, 6))
sns.barplot(x=labels, y=values, palette=["green", "orange"])
plt.title("Distribución de Predicciones por Nivel de Confianza")
plt.ylabel("Número de Imágenes")
for i, v in enumerate(values):
    plt.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.show()

# ============================
# 18. OBTENER PREDICCIONES
# ============================

#Generamos otro df con más información
y_prob = model.predict(test_generator, verbose=1)      # probabilidades softmax
y_pred = np.argmax(y_prob, axis=1)                     # clase top-1 predicha
confianza_top1 = np.max(y_prob, axis=1)                # probabilidad asociada

y_true = test_generator.classes

acierto = (y_pred == y_true)

df_resultados = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred,
    'confianza_top1': confianza_top1,
    'acierto': acierto
})

# ============================
# 19. CALCULAR MÉTRICAS POR UMBRAL
# ============================
umbrales = np.linspace(0, 1, 500)

aciertos = df_resultados[df_resultados['acierto']]
errores  = df_resultados[~df_resultados['acierto']]

res = []
for u in umbrales:
    aciertos_conf = (aciertos['confianza_top1'] >= u).sum()
    errores_conf  = (errores['confianza_top1']  >= u).sum()

    tasa_aciertos = aciertos_conf / len(aciertos) if len(aciertos) > 0 else 0
    tasa_errores  = errores_conf / len(errores) if len(errores) > 0 else 0

    res.append({
        'umbral': u,
        'tasa_aciertos': tasa_aciertos,
        'tasa_errores': tasa_errores
    })

df_umbral = pd.DataFrame(res)

# ============================
# 20. GRAFICAR TRADE-OFF
# ============================
#Graficamos cómo cambian aciertos y errores retenidos al variar el umbral de confianza
plt.figure(figsize=(10, 6))
plt.plot(df_umbral['umbral'], df_umbral['tasa_aciertos'], 
         label='Aciertos retenidos', color='green')
plt.plot(df_umbral['umbral'], df_umbral['tasa_errores'], 
         label='Errores retenidos', color='red')
plt.xlabel('Umbral de confianza top-1')
plt.ylabel('Proporción retenida')
plt.title('Trade-off entre aciertos y errores al variar el umbral (Versión 5)')
plt.legend()
plt.grid(True)
plt.show()

# ============================
# 21. CÁLCULAMOS EL UMBRAL ÓPTIMO
# ============================
#Calculamos score y umbral óptimo
df_umbral['score'] = df_umbral['tasa_aciertos'] - df_umbral['tasa_errores']
umbral_optimo = df_umbral.loc[df_umbral['score'].idxmax(), 'umbral']
print(f"📌 Umbral óptimo encontrado: {umbral_optimo:.4f}")

#Filtramos predicciones con confianza >= umbral óptimo
automaticas = df_resultados[df_resultados['confianza_top1'] >= umbral_optimo]

#Contamos resultados
total_automaticas = len(automaticas)
aciertos_automaticos = automaticas['acierto'].sum()
errores_automaticos = total_automaticas - aciertos_automaticos

#Porcentajes
cobertura = (total_automaticas / len(df_resultados)) * 100
porcentaje_error = (errores_automaticos / total_automaticas * 100) if total_automaticas > 0 else 0

print(f"📦 Total de imágenes evaluadas en test: {len(df_resultados)}")
print(f"✅ Imágenes aceptadas automáticamente: {total_automaticas} ({cobertura:.2f}%)")
print(f"   - Aciertos automáticos: {aciertos_automaticos}")
print(f"   - Errores automáticos: {errores_automaticos}")
print(f"❌ Porcentaje de error en automáticas: {porcentaje_error:.2f}%")

#Graficamos los datos para su análisi
cobertura_list = []
error_list = []

for u in df_umbral['umbral']:
    auto = df_resultados[df_resultados['confianza_top1'] >= u]
    total_auto = len(auto)
    if total_auto > 0:
        errores_auto = (~auto['acierto']).sum()
        porcentaje_error = errores_auto / total_auto * 100
        cobertura = total_auto / len(df_resultados) * 100
    else:
        porcentaje_error = 0
        cobertura = 0
    
    cobertura_list.append(cobertura)
    error_list.append(porcentaje_error)

df_umbral['cobertura'] = cobertura_list
df_umbral['porcentaje_error'] = error_list

fig, ax1 = plt.subplots(figsize=(10,6))

ax1.plot(df_umbral['umbral'], df_umbral['cobertura'], label="Cobertura (%)", color='blue')
ax1.set_xlabel("Umbral de confianza", color='black')
ax1.set_ylabel("Cobertura (%)", color='black')
ax1.tick_params(axis='y', colors='black')
ax1.tick_params(axis='x', colors='black')

ax2 = ax1.twinx()
ax2.plot(df_umbral['umbral'], df_umbral['porcentaje_error'], label="Error (%)", color='red')
ax2.set_ylabel("Error en automáticas (%)", color='black')
ax2.tick_params(axis='y', colors='black')

ax1.axvline(umbral_optimo, color='green', linestyle='--', label=f"Umbral óptimo {umbral_optimo:.3f}")

fig.suptitle("Cobertura vs Error al variar el umbral", fontsize=14, color='black')
fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
plt.tight_layout()
plt.show()

# ============================
# 22. CÁLCULAMOS LAS FRUTAS MÁS CONFUNDIDAS ENTRE SÍ
# ============================

#Matriz de confusión sin normalizar (números de imágenes)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))

#Extraemos confusiones (fuera de la diagonal)
confusiones = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i, j] > 0:  # solo errores
            confusiones.append({
                "real": label_encoder.classes_[i],
                "predicho": label_encoder.classes_[j],
                "cantidad": cm[i, j]
            })

# Pasamos el DataFrame ordenado
df_confusiones = pd.DataFrame(confusiones).sort_values(by="cantidad", ascending=False)

print("🔍 Principales confusiones entre frutas:")
print(df_confusiones.head(10))

# Generamos un gráfico de barras con las confusiones más comunes
plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_confusiones.head(10),
    x="cantidad", y="real", hue="predicho", dodge=False
)
plt.title("Frutas más confundidas entre sí")
plt.xlabel("Número de imágenes confundidas")
plt.ylabel("Fruta real")
plt.legend(title="Predicha como")
plt.tight_layout()
plt.show()







