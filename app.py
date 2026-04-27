from flask import Flask, render_template, request, jsonify
import linearRegression
import RegresionLogistica
import Clustering

# NUEVO
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route("/linearRegression/", methods=["GET", "POST"])
def calculateGrade():
    result = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        result = linearRegression.calculateGrade(hours)
    return render_template("linearRegression.html", result=result)


@app.route("/logisticRegression/", methods=["GET", "POST"])
def logisticPrediction():
    result = None
    if request.method == "POST":
        result = RegresionLogistica.predict(request.form)
    return render_template("logisticRegression.html", result=result)


@app.route("/clustering/")
def clustering():
    data = Clustering.applyClustering()
    return render_template("clustering.html", data=data)



@app.route("/unsupervised/")
def unsupervised_home():
    return render_template("unsupervised/index.html")

@app.route("/unsupervised/concepts")
def unsupervised_concepts():
    return render_template("unsupervised/basic_concepts.html")

@app.route("/unsupervised/kmeans-manual")
def kmeans_manual():
    return render_template("unsupervised/kmeans_manual.html")

@app.route("/unsupervised/clustering")
def unsupervised_clustering():
    return render_template("unsupervised/clustering_app.html")


@app.route("/api/kmeans")
def api_kmeans():
    data = pd.DataFrame({
        "x": np.random.rand(1000),
        "y": np.random.rand(1000)
    })

    model = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = model.fit_predict(data)

    centroids = model.cluster_centers_.tolist()

    return jsonify({
        "data": data.to_dict(orient="records"),
        "centroids": centroids
    })

# ========================
if __name__ == '__main__':
    app.run(debug=True)