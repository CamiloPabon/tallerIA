import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# 1. Cargar el dataset
housing_data = pd.read_csv('dataset/housing.csv')

# 2. Tratamiento de los datos: eliminar la columna 'Address' ya que no es relevante
housing_data_cleaned = housing_data.drop(columns=['Address'])

# 3. Gráfico de correlación entre las variables
plt.figure(figsize=(10, 8))
sns.heatmap(housing_data_cleaned.corr(), annot=True, cmap="YlGnBu")
plt.title('Mapa de Correlación')
plt.show()

# 4. Gráfico de pairplots para determinar la relación entre las características
sns.pairplot(housing_data_cleaned)
plt.show()

# Separar las características (X) y la variable objetivo (y)
X = housing_data_cleaned.drop('Price', axis=1)
y = housing_data_cleaned['Price']

# 5. Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Crear el modelo de regresión lineal
linreg = LinearRegression()

# Entrenar el modelo
linreg.fit(X_train_scaled, y_train)

# 7. Hacer predicciones en el conjunto de prueba y calcular el error cuadrático medio
y_pred = linreg.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio: {mse}')

# 8. Calcular el score del modelo
score = linreg.score(X_test_scaled, y_test)
print(f'Score del modelo: {score}')

# 9. Validación cruzada (cross-validation)
cross_val_scores = cross_val_score(linreg, X_train_scaled, y_train, cv=5)
print(f'Resultados de la Validación Cruzada: {cross_val_scores}')
print(f'Promedio de la Validación Cruzada: {cross_val_scores.mean()}')

# 10. Exportar el modelo entrenado y el escalador en el directorio actual
with open('models/linear_regression_housing_model.pkl', 'wb') as model_file:
    pickle.dump(linreg, model_file)

with open('models/scaler_housing.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)


