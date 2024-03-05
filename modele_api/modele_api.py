from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage for TensorFlow
from PIL import Image
import io
from pydantic import BaseModel

import sys
sys.path.append('../')
from config import MLFLOW_SERVER, MLFLOW_MODEL_RUNS, HOST, PORT_API_MODEL

mlflow.set_tracking_uri(uri=MLFLOW_SERVER)

app = FastAPI()

logged_model = MLFLOW_MODEL_RUNS

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
print(loaded_model)


# Fonction pour normaliser une image
# def normalize_image(image):
#     # Convertir l'image en niveaux de gris
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Redimensionner l'image à une taille fixe (si nécessaire)
#     resized_image = cv2.resize(gray_image, (224, 224))
#     # Convertir l'image en niveaux de gris en une image avec trois canaux de couleur
#     color_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
#     # Normaliser les valeurs de pixel pour être dans la plage [0, 1]
#     normalized_image = color_image / 255.0
#     # Remodeler l'image pour être dans le format attendu par le modèle
#     normalized_image = np.expand_dims(normalized_image, axis=0)
#     return normalized_image

def normalize_image(img, target_size):
    if len(img.shape) == 3:
        # Convertir en niveaux de gris si ce n'est pas déjà le cas
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Appliquer un filtre pour supprimer le bruit (par exemple, un filtre gaussien)
    denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Détecter les contours pour trouver le crop optimal
    _, thresh = cv2.threshold(denoised_img, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Trouver le contour avec la plus grande aire
        max_contour = max(contours, key=cv2.contourArea)

        # Obtenir les coordonnées du rectangle englobant
        x, y, w, h = cv2.boundingRect(max_contour)

        # Cropper l'image pour obtenir la région d'intérêt
        cropped_img = img[y:y+h, x:x+w]

        # Redimensionner à target_size (pour s'assurer que toutes les images ont la même taille)
        normalized_image = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
    else:
        # Redimensionner à target_size si aucun contour n'est détecté
        normalized_image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    if len(normalized_image.shape) != 3:
        # Convertir en couleur si besoin
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
    return normalized_image

def configure_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Autorise toutes les origines
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Autorise les méthodes HTTP spécifiées
        allow_headers=["*"],  # Autorise tous les en-têtes
    )
configure_cors(app)

# Route feedback

class Feedback(BaseModel):
    # Ajoutez les champs nécessaires pour les prédictions
    file: str = ""
    comment: str = ""
    predict_score: float = None

@app.post("/feedback/")

async def feedback(data: Feedback):
    print(data)
    # print(file, comment, predict_score)


# Route pour effectuer des prédictions
@app.post("/predict/")

async def predict(file: UploadFile):
    # # Lire l'image depuis le fichier téléchargé
    # contents = await file.read()
    # nparr = np.frombuffer(contents, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # # Normaliser l'image
    # # target_size = (224, 224)
    # normalized_image = normalize_image(image)
    
    # # Faire une prédiction avec le modèle
    # prediction = loaded_model.predict(normalized_image)
    # print(prediction)
    # # Retourner la prédiction
    # prediction_result = prediction.tolist()  # Si result est un tableau numpy
    # print(prediction_result)
    # return {"prediction": prediction_result}
        # Lire les données du fichier UploadFile
        # contents = await file.read()
        # Convertir les données en un objet image avec PIL
        file = io.BytesIO(await file.read())
        image = Image.open(file)
        X_img = np.array(image)
        
        img = normalize_image(X_img, (224, 224))
        img = np.expand_dims(img, axis=0)
        result = loaded_model.predict(img)
        # Convertir le résultat en un type de données sérialisable en JSON
        prediction_result = result.tolist()  # Si result est un tableau numpy
        print(prediction_result)
        # Extraire la probabilité de la tumeur
        tumor_probability = prediction_result[0][0]
        if tumor_probability < 0.5 :
            label = 'Tumeur NO'
        else:
            label = 'Tumeur YES'
        return tumor_probability, label


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT_API_MODEL)