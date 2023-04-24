# Model Trainning
Repositorio para poder relanzar el entrenamiento del modelo cada X tiempo

# Docker:

## Local
Se añadido el docker-compose (poisblemente sea necesario instalar el paquete) porque así se se puede montar la aplicación web y cualquier otro servicio de forma local

### Construir la imagen
    docker-compose build

### Ejecutar el contenedor
    docker-compose up
    
# GCP
Los depliegues han sido automatizados con Cloud Build


# CloudBuild
Se crea el fichero cloudbuild.yaml con los pasos del despliegue. Para que funcione correctamente hay que declarar dentro del trigger de despliegue las siguientes variables de entorno:
  _BUCKET: Nombre del bucket en el que se almacenaran los modelos de la aplicación
  
Se puede usar como ejemplo las variables definidas dentro del cloudbuild.yml

# Swagger (Documentación API)
Se ha montado dentro del proyecto un API con Swagger totalmente operativa. Se encuentra activa dentro de la ruta /docs

También se dispone la documentación en ReDoc en la ruta /redoc

# Documentacion
1º Subida fotos Truefaces/Dataface y truefaces/Datanoface
2º Añadido la cuenta se servicio 
3º Subida del modelo modificado para su ejecucion en Cloud Run : train_model.py
4º Añadido codigo en app/routes/train.py   ejecucion entrenamiento modelo.
5ª añadido triguer al YAML del cloudbuild. Este archivo YAML define dos pasos en el proceso de compilación y un disparador 
que programará un trabajo de entrenamiento de modelo cada tres meses y su guardado en el bucket.
