import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Cargar el dataset
heart_data = pd.read_csv('dataset/heart.csv')

# Separar las características (X) y la variable objetivo (y)
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# 2. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Crear el modelo de regresión logística
logreg = LogisticRegression()

# Entrenar el modelo
logreg.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = logreg.predict(X_test_scaled)

# 4. Calcular el score del modelo
score = accuracy_score(y_test, y_pred)
print(f'Score del modelo: {score}')

# 5. Validación cruzada (cross-validation)
cross_val_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)
print(f'Resultados de la Validación Cruzada: {cross_val_scores}')
print(f'Promedio de la Validación Cruzada: {cross_val_scores.mean()}')

# 6. Exportar el modelo entrenado y el escalador en el directorio actual
with open('models/logistic_regression_heart_model.pkl', 'wb') as model_file:
    pickle.dump(logreg, model_file)

with open('models/scaler_heart.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
