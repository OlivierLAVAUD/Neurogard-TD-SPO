# Application Neuroguard

## Objectif de cette application

Cette application est une application médicale qui a pour objectif de gérer des scanners de patients et de prédire la présence d'une tumeur. Cette prédiction est ensuite confirmée ou non par un expert.

app.py est une application FastAPI qui gère les opérations CRUD (Create, Read, Update, Delete) sur une base de données MongoDB pour gérer des informations sur des patients.
On utilise Pydantic pour la validation des données

## Structure du projet

```bash
project/
│
├── braintumor-ui/
│   ├── static/    
│   |   └── styles.css          #style des pages html
│   |
│   ├── templates/
│   │   ├──add_patient.html     #page HTML pour ajouter le patient
│   │   ├──add_validation.html  #page HTML pour valider la prédiction par l'expert
│   │   ├──edit_patient.html    #page HTML pour ajouter le patient 
│   │   ├──view_patient.html    #page HTML visualiser un patient et imprimer le rapport
│   │   └──view_patients.html   #page générale avec la vue de tous les patients 
│   │   
│   ├── __init__.py   
│   │   
|   └── app.py                  # Fichier principale de l'app
|
|
├── modele-api/
│   ├── modele_api.py       # API du modele de prédiction
│   │
│   └── process-data.ipynb  # Pour générer et entrainer un modele
│
├── .gitattributes
├── .gitignore
│
├── config.py               # A générer avec les paramètres propre à l'utilisateur
├── config.py.example       # A dupliquer pour créer config.py
│
└── README.md
```

## Requirement:
voir requirements.txt

## Démarrer l'app:

1 - démarrer un serveur mlflow par exemple sur le port PORT_API_MODEL

```
mlflow server --host 127.0.0.1 --port PORT_API_MODEL
```

2 - executer le notebook process-data.ipynb
cela permet d'entrainer un modèle
Se connecter à mlFlow et récupérer l'id du runs

3 - dupliquer le fichier config.py.example et le renommer config.py
dans ce fichier saisir le port mlflow choisi, le runs du modele créé précedemment dans mlflow, les paramètres MongoDB et le port choisi pour l'app

4 - Démarrer l'api du modèle
Se placer dans le repertoire modele_api.py et executer la commande :

```
python modele_api.py
```

5 - Démarrer l'application
Se placer dans le repertoire braintumor-ui et executer la commande :

```
python app.py
```

6 - se rendre sur la page http://127.0.0.1:{PORT_APP}/view_patients
