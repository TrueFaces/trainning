from fastapi import APIRouter, HTTPException, UploadFile, File
from app.utils.storage import upload_file_to_bucket, delete_file_from_bucket
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from google.cloud import storage
from typing import List
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import time

router = APIRouter(prefix="/train", tags=["train"])


@router.post("/")
async def train_model(bucket_name, folder1, folder2, img_size=100, test_size=0.2):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    data = []
    labels = []

    for category in bucket.list_blobs(prefix=folder1):
        img_path = f"gs://{bucket_name}/{category.name}"
        blob = bucket.blob(category.name)
        img_bytes = blob.download_as_bytes()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a espacio de color RGB
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32') / 255.0  # Normalizar la imagen
            data.append(img)
            labels.append(0)
        else:
            print(f"No se pudo cargar la imagen: {img_path}")

    for category in bucket.list_blobs(prefix=folder2):
        img_path = f"gs://{bucket_name}/{category.name}"
        blob = bucket.blob(category.name)
        img_bytes = blob.download_as_bytes()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a espacio de color RGB
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32') / 255.0  # Normalizar la imagen
            data.append(img)
            labels.append(1)
        else:
            print(f"No se pudo cargar la imagen: {img_path}")

    data = np.array(data).reshape(-1, img_size, img_size, 3)  # Reformatear a un array 4D
    labels = np.array(labels)

    #print(f"Se cargaron {len(data)} imágenes.")

    # Dividir los datos y las etiquetas en conjuntos de entrenamiento y validación
    X_train_val, X_test, y_train_val, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size, random_state=42)

    return X_test, y_test, X_train, y_train, X_val, y_val


@router.post("/start")
async def start_train_model(bucket_name: str, folder1: str, folder2: str, img_size: int = 100, test_size: float = 0.2):
    # Llamar a la función train_model con los parámetros proporcionados
    X_test, y_test, X_train, y_train, X_val, y_val = await train_model(bucket_name, folder1, folder2, img_size, test_size)

    # Cargar el modelo pre-entren


    # Cargar el modelo pre-entrenado VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    # Agregar una nueva capa superior al modelo para clasificación binaria
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compilar el modelo con pérdida de entropía cruzada binaria y optimizador Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Entrenar el modelo con 15 épocas y un tamaño de lote de 100
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    H = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=100, callbacks=[early_stop])
    
    # Guardar el modelo entrenado en Google Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob('truefaces/model/modelo_prueba.h5')
    model.save(blob)
    
    return {"message": "Model trained and saved successfully!"}
