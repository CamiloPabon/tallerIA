from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Cargar el modelo y el escalador desde los archivos pickle
file_name_modelo = "models/linear_regression_housing_model.pkl"
file_name_scaler = "models/scaler_housing.pkl"

with open(file_name_modelo, 'rb') as archivo_lectura:
    model_loaded = pickle.load(archivo_lectura)

with open(file_name_scaler, 'rb') as archivo_salida:
    scaler_loaded = pickle.load(archivo_salida)


# Definir la ruta para la predicción
@app.post('/houses')
async def predict(data: dict):
    # Convertir los datos a un DataFrame
    df = pd.DataFrame([data])

    # Aplicar el escalador a los datos
    scaled_data = scaler_loaded.transform(df)

    # Realizar la predicción con el modelo cargado
    predict = model_loaded.predict(scaled_data)

    # Devolver la predicción como una lista
    return {'predict': predict.tolist()}


# Ruta de prueba
@app.get("/")
async def root():
    return {"message": "Servicio de Predicción de Precios de Casas"}


# Ruta de saludo personalizado
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
