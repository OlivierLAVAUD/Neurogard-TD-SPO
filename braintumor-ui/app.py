import base64
import requests
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel
import json

import sys

sys.path.append("../")
from config import HOST, PORT_APP, MONGO_SERVER, MONGO_DB, PORT_API_MODEL

app = FastAPI()

# Connexion à la base de données MongoDB
client = MongoClient(MONGO_SERVER)
db = client[
    MONGO_DB
]  # Remplacez "your_database_name" par le nom de votre base de données MongoDB


# Modèle Pydantic pour les données du patient
class PatientModel(BaseModel):
    name: str
    age: int
    gender: str
    radio: str = ""
    predict_score: float = 0.0
    predict_label: str = ""
    validation: str = ""
    comment: str = ""
    status: int = 0


# Modèles Pydantic pour la modification du patient
class PatientUpdateModel(BaseModel):
    name: str
    age: int
    gender: str
    radio: str = ""
    predict_score: float = 0.0
    predict_label: str = ""
    validation: str = ""
    comment: str = ""
    status: int = 0


# Modèles Pydantic pour la visualisation des patients
class PatientViewModel(BaseModel):
    name: str
    age: int
    gender: str
    id: str
    radio: str = ""
    predict_score: float = 0.0
    predict_label: str = ""
    validation: str = ""
    comment: str = ""
    status: int = 0


class ValidationModel(BaseModel):
    # Ajoutez les champs nécessaires pour les prédictions
    comment: str = ""


# Modèle Pydantic pour les prédictions (à adapter selon vos besoins)
class PredictionModel(BaseModel):
    # Ajoutez les champs nécessaires pour les prédictions
    pass


# Montez le répertoire 'static' pour servir les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")


# Instance du moteur de modèles Jinja2 pour la gestion des templates HTML
templates = Jinja2Templates(directory="templates")


# Route pour ajouter un patient
@app.get("/add_patient", response_class=HTMLResponse)
def add_patient(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})


@app.post("/add_patient")
async def add_patient_post(patient: PatientModel):
    if patient.radio != "":
        files = {"file": base64.b64decode(patient.radio)}
        # Make a POST request to the FastAPI endpoint
        response = requests.post(
            f"http://{HOST}:{PORT_API_MODEL}/predict/", files=files
        )
        if response.status_code == 200:
            print(f"response", json.loads(response.text))
            response_data = json.loads(response.text)
            patient.predict_score = round(response_data[0], 3)
            patient.predict_label = response_data[1]
        else:
            print(f"error", response)
    # Insérer le patient dans la base de données
    patient_data = patient.dict()
    db.patients.insert_one(patient_data)
    return JSONResponse(content={"redirect_url": "/view_patients"})


# Route pour visualiser tous les patients
@app.get("/view_patients", response_class=HTMLResponse)
async def view_patients(request: Request):
    # Récupérer tous les patients depuis la base de données
    patients = [
        PatientViewModel(id=str(patient["_id"]), **patient)
        for patient in db.patients.find().sort("predict_score", -1)
    ]
    return templates.TemplateResponse(
        "view_patients.html", {"request": request, "patients": patients}
    )


# Route pour éditer un patient
@app.get("/edit_patient/{patient_id}", response_class=HTMLResponse)
async def edit_patient(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
  #  status = PatientModel(**db.patients.find_one({"_id": ObjectId(status)}))
    return templates.TemplateResponse(
        "edit_patient.html",
        {"request": request, "patient": patient, "patient_id": patient_id},
    )


@app.post("/edit_patient/{patient_id}")
async def edit_patient_post(patient_id: str, patient: PatientUpdateModel):
    if patient.radio != "":
        files = {"file": base64.b64decode(patient.radio)}
        # Make a POST request to the FastAPI endpoint
        response = requests.post(
            f"http://{HOST}:{PORT_API_MODEL}/predict/", files=files
        )
        if response.status_code == 200:
            print(f"response", json.loads(response.text))
            response_data = json.loads(response.text)
            patient.predict_score = round(response_data[0], 3)
            patient.predict_label = response_data[1]
            
        else:
            print(f"error", response)
    # Mettre à jour le patient dans la base de données
    db.patients.update_one(
        {"_id": ObjectId(patient_id)}, {"$set": patient.model_dump()}
    )
    return RedirectResponse(url="/view_patients")


# Route pour voir un patient
@app.post("/view_patient/{patient_id}")
async def view_patient_post(patient_id: str, patient: PatientUpdateModel):
    # Mettre à jour le patient dans la base de données
    db.patients.update_one(
        {"_id": ObjectId(patient_id)}, {"$set": patient.model_dump()}
    )
    return RedirectResponse(url="/view_patients")


# Route pour voir un patient
@app.get("/view_patient/{patient_id}", response_class=HTMLResponse)
async def view_patient(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
 #   status = PatientModel(**db.patients.find_one({"_id": ObjectId(status)}))
    return templates.TemplateResponse(
        "view_patient.html",
 #       {"request": request, "patient": patient, "patient_id": patient_id, "status": status},
         {"request": request, "patient": patient, "patient_id": patient_id, "status": status},
    )


# Route pour valider le dossier patient
@app.post("/add_validation/{patient_id}/{validation}")
async def add_validation_post(
    patient_id: str, validation: bool, formData: ValidationModel
):
    # Mettre à jour le statut du patient dans la base de données
    valid = "true" if validation else "false"
    # if validation: status = 2
    status = 2

    db.patients.update_one(
        {"_id": ObjectId(patient_id)},
        {"$set": {"validation": valid, "status": status, "comment": formData.comment}},
    )

    if valid == "false":

        patient_data = PatientModel(
            **db.patients.find_one({"_id": ObjectId(patient_id)})
        )

        # Envoi d'une notification à l'API modèle en cas de divergence :
        data_divergence = {
            "image": patient_data.radio,
            "prediction": patient_data.predict_score,
            "avis_expert": patient_data.comment,
        }
        # Make a POST request to the FastAPI endpoint
        response = requests.post(
            f"http://{HOST}:{PORT_API_MODEL}/feedback/", json=data_divergence
        )
        if response.status_code == 200:
            print("Feedback envoyé avec succès.")
            print(f"API modèle :", json.loads(response.text))
        else:
            print(f"error", response)

    return RedirectResponse(url="/view_patients")


# Route pour valider le dossier patient
@app.get("/add_validation/{patient_id}", response_class=HTMLResponse)
async def add_validation(request: Request, patient_id: str):
    # Récupérer les informations du patient pour affichage dans le formulaire
    patient = PatientModel(**db.patients.find_one({"_id": ObjectId(patient_id)}))
    return templates.TemplateResponse(
        "add_validation.html",
        {"request": request, "patient": patient, "patient_id": patient_id},
    )


# Route pour supprimer un patient
@app.post("/delete_patient/{patient_id}")
async def delete_patient(patient_id: str):
    # Vérifiez si le patient existe
    patient = db.patients.find_one({"_id": ObjectId(patient_id)})
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    # Supprimez le patient de la base de données
    db.patients.delete_one({"_id": ObjectId(patient_id)})
    return {"message": "Patient deleted successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT_APP)
