import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score

# Cargar el archivo CSV con los logs de Nginx
df_logs = pd.read_csv('logs/nginx_access_log.csv')

# Definir X (variables independientes) y y (variable dependiente)
X = df_logs[['requests_per_second', 'size', 'status']]
y = df_logs['is_dos_attack']

# Escalar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Crear el modelo AdaBoost con un árbol de decisión débil como estimador base
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo en los datos de prueba
test_score = model.score(X_test, y_test)
print(f"Score del modelo en los datos de prueba: {test_score}")

# Realizar validación cruzada con 5 particiones
cross_val_results = cross_validate(model, X_scaled, y, cv=5, scoring=make_scorer(accuracy_score))
mean_cross_val_score = cross_val_results['test_score'].mean()
print(f"Promedio de score con validación cruzada (5 particiones): {mean_cross_val_score}")

# Guardar el modelo entrenado en un archivo .pkl
with open("models/AdaBoostClassifier.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Guardar el scaler también en un archivo .pkl
with open("models/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

