from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd

import logging

# Inicializar la app de FastAPI y cargar el modelo y el scaler
app = FastAPI()
import pickle

# Cargar el modelo y el scaler desde los archivos pickle
def load_model():
    try:
        with open("fastApiDOS/models/AdaBoostClassifier.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("fastApiDOS/models/scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {str(e)}")

# Configurar el logging  para que se muestren los mensajes de INFO
logging.basicConfig(level=logging.INFO)

# Cargar el modelo preentrenado y el scaler
model, scaler = load_model()

# Definir la estructura de los datos que vamos a recibir
class DosAttackRequest(BaseModel):
    requests_per_second: float
    size: float
    status: int

    # Validar que las variables numéricas sean positivas
    @validator('requests_per_second', 'size')
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Debe ser un número positivo')
        return v

@app.post("/DOS", summary="Detección de ataque DOS", description="Detecta si una solicitud es un ataque DOS basado en la cantidad de peticiones por segundo, el tamaño y el estado.")
def predict_dos_attack(request: DosAttackRequest):
    try:
        # Convertir los datos de la solicitud a un DataFrame
        user_data = pd.DataFrame({
            'requests_per_second': [request.requests_per_second],
            'size': [request.size],
            'status': [request.status]
        })

        # Escalar los datos de entrada usando el scaler entrenado
        user_data_scaled = scaler.transform(user_data)

        # Realizar la predicción
        prediction = model.predict(user_data_scaled)

        # Loggear la predicción
        logging.info(f"Predicción realizada: {prediction[0]}")

        # Devolver el resultado de la predicción
        if prediction[0] == 1:
            return {"message": "La solicitud es considerada un ataque DOS."}
        else:
            return {"message": "La solicitud es normal."}

    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {str(e)}")