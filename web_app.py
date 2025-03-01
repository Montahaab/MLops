from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:5000/predict"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            features = request.form["features"]
            features_list = list(map(float, features.split(",")))

            response = requests.post(API_URL, json={"features": features_list})

            # Vérifier si la requête a réussi (status code 200)
            if response.status_code != 200:
                error = "Erreur lors de la requête à l'API."
            else:
                data = response.json()
                prediction = data.get("prediction", "Erreur dans la réponse de l'API")

        except requests.exceptions.JSONDecodeError:
            error = "Réponse invalide de l'API."
        except requests.exceptions.RequestException:
            error = "Impossible de se connecter à l'API."

    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True,port=5001)

