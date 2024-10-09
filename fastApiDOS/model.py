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
