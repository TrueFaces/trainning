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
async def start_train_model():
    
    def etiquetar_imagenes(bucket_name, folder1, folder2, img_size=100, test_size=0.2):
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

# Cargar los datos
bucket_name = "truefaces"
folder1 = "Datanoface/"
folder2 = "Dataface/"
X_test, y_test, X_train, y_train, X_val, y_val = etiquetar_imagenes(bucket_name, folder1, folder2)

# Load the pre-trained VGG16 model without the top layer (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze the weights of the pre-trained layers so they are not updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add a new top layer to the model for binary classification
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile the model with binary cross-entropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for 10 epochs with a batch size of 32
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
H = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=100, callbacks=[early_stop])

# Save the trained model to Google Cloud Storage
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blob = bucket.blob('truefaces/model/modelo_prueba.h5')
model.save(blob)


@router.post("/upload")
async def upload_model(files: List[UploadFile] = File(...)):
    for file in files:
        filename = f"{int(time.time())}_{file.filename}"
        path = await upload_file_to_bucket("test_models", filename, file)

    return {"status": "ok"}

@router.delete("/remove/{filename}")
async def remove_model(filename: str):
    await delete_file_from_bucket(filename)

    return {"status": "ok"}
