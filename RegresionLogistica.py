
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 2. Carga de datos
data = pd.read_csv('dataset_regresion_logistica.csv')
print(data.head())
print(data.info())
print(data.describe())

# 3. Preparación de datos
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entrenamiento
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# 5. Predicciones
y_pred = logistic_model.predict(X_test_scaled)

# 6. Evaluación
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 7. Reporte y exactitud
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {accuracy * 100:.2f}%')

def predict(form):
    entrada = np.array([[
        float(form["edad"]),
        float(form["ingreso_mensual"]),
        float(form["visitas_web_mes"]),
        float(form["tiempo_sitio_min"]),
        float(form["compras_previas"]),
        float(form["descuento_usado"])
    ]])
    entrada_scaled = scaler.transform(entrada)
    prediccion = logistic_model.predict(entrada_scaled)
    return "Comprará" if prediccion[0] == 1 else "No comprará"