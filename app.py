from flask import Flask, render_template, request
import linearRegression
import RegresionLogistica

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