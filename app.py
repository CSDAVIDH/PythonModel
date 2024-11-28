from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Cargar el modelo guardado
model = joblib.load("model.joblib")

# Crear la aplicación Flask
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return render_template("pindex.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener los datos de entrada del JSON recibido
        input_data = request.get_json()

        # Validar que las claves necesarias estén en los datos
        required_keys = [
            "sq_mt_built",
            "n_rooms",
            "n_bathrooms",
            "has_lift",
            "has_parking",
            "house_type_id",
        ]
        for key in required_keys:
            if key not in input_data:
                return jsonify({"error": f"Falta la clave requerida: {key}"}), 400

        # Convertir los datos de entrada en un array para el modelo
        input_features = np.array(
            [
                [
                    input_data["sq_mt_built"],
                    input_data["n_rooms"],
                    input_data["n_bathrooms"],
                    input_data["has_lift"],
                    input_data["has_parking"],
                    input_data["house_type_id"],
                ]
            ]
        )

        # Realizar la predicción
        prediction = model.predict(input_features)

        # Retornar la predicción como respuesta JSON
        return jsonify({"predicted_price": prediction[0]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
