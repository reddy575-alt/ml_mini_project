import numpy as np
from flask import Flask, render_template, request
import pickle
import warnings

warnings.filterwarnings("ignore")



with open("scaled.pkl", "rb") as sc:
    scale = pickle.load(sc)



with open("model.pkl", "rb") as r:
    reg = pickle.load(r)



app = Flask(__name__)



@app.route("/")
def main_page():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get values from frontend IN CORRECT ORDER
        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        thalach = float(request.form["thalach"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        thal = float(request.form["thal"])

        data = np.array([[
            age,
            sex,
            cp,
            thalach,
            oldpeak,
            slope,
            thal
        ]])

        # Scale the data
        scaled_data = scale.transform(data)

        # Predict using SCALED data
        prediction = reg.predict(scaled_data)[0]

        # Send prediction to HTML
        return render_template(
            "index.html",
            prediction=int(prediction)
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )

if __name__ == "__main__":
    app.run(debug=True)