from fastapi import APIRouter, HTTPException, UploadFile
from app.utils.storage import upload_file_to_bucket, delete_file_from_bucket
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from google.cloud import storage
from typing import List
from app.train_model import etiquetar_imagenes

router = APIRouter(prefix="/train", tags=["train"])

@router.post("/")
async def start_train_model():
   
    # Load the data
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
    x = Flatten()(base_model.output)
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
    blob = bucket.blob('truefaces/model/modelo.h5')
    model.save(blob)

    # Return the training history
    return {"history": H.history}

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
