import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ✅ función para entrenar
def train_model():
    data = pd.read_csv('dataset_regresion_logistica.csv')

    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return model, scaler


# ✅ entrenar UNA vez
logistic_model, scaler = train_model()


# ✅ función que usa Flask
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