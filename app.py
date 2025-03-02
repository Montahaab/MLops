from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API FastAPI ! Utilisez /docs pour voir la documentation."
    }


# Charger le modèle et le scaler
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le fichier modèle '{MODEL_PATH}' est introuvable.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Le fichier scaler '{SCALER_PATH}' est introuvable.")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class Features(BaseModel):
    features: List[float]


@app.post("/predict")
def predict(data: Features):
    try:
        # Conversion des données en numpy array
        features = np.array(data.features).reshape(1, -1)

        # Normalisation des données d'entrée
        features_scaled = scaler.transform(features)

        # Prédiction
        prediction = model.predict(features_scaled)

        return {"prediction": int(prediction[0])}
    except Exception:
        return {"error": "Une erreur est survenue pendant la prédiction."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
