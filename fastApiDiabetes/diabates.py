import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Cargar el dataset con pandas
diabetes_data = pd.read_csv('dataset/diabetes.csv')

# 2. Tratamiento y adecuación de los datos

# Separar las características (X) y la variable objetivo (y)
X = diabetes_data.drop('Salida', axis=1)
y = diabetes_data['Salida']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Crear el modelo de regresión logística y validación cruzada
logreg = LogisticRegression()

# Realizar validación cruzada con 5 particiones
cross_val_scores = cross_val_score(logreg, X_train_scaled, y_train, cv=5)

# Entrenar el modelo
logreg.fit(X_train_scaled, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = logreg.predict(X_test_scaled)

# 4. Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del Modelo: {accuracy}')

# Mostrar los resultados de la validación cruzada
print(f'Validación Cruzada (5 particiones): {cross_val_scores}')
print(f'Promedio de la Validación Cruzada: {cross_val_scores.mean()}')

# 5. Predicción de ejemplo usando datos de prueba
example_data = X_test_scaled[0].reshape(1, -1)
example_prediction = logreg.predict(example_data)
print(f'Ejemplo de Predicción (0=No diabetes, 1=Diabetes): {example_prediction}')

# Exportar el modelo entrenado usando pickle
with open('models/logistic_regression_diabetes_model.pkl', 'wb') as model_file:
    pickle.dump(logreg, model_file)

# Exportar el escalador
with open('models/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
