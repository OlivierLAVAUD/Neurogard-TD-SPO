import subprocess
import base64
import os

# Liste des packages à vérifier
packages = [
    "requests",
    "uvicorn",
    "fastapi",
    "pymongo",
    "pydantic",
    "mlflow",
    "opencv-python-headless",
    "pillow",
    "numpy",
    "keras",
    "matplotlib",
    "pandas",
    "tensorflow"
]

# Créer le fichier requirements.txt
with open("requirements.txt", "w") as f:
    for package in packages:
        # Utiliser pip show pour obtenir les informations sur le package
        result = subprocess.run(["pip", "show", package], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        version_lines = [line for line in lines if line.startswith("Version:")]
        if version_lines: # Vérifier que la liste n'est pas vide
            version_line = version_lines[0]
            version = version_line.split(": ")[1]
            f.write(f"{package}=={version}\n")
        else:
            print(f"Version information not found for {package}.")

# Encoder le fichier requirements.txt en base64
with open("requirements.txt", "rb") as f:
    encoded_string = base64.b64encode(f.read()).decode('utf-8')

# Sauvegarder l'encodage en base64 dans un fichier
with open("requirements_base64.txt", "w") as f:
    f.write(encoded_string)

print("Fichier requirements_base64.txt créé avec succès.")

