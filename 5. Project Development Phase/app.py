from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Store values to keep them in form
        values = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }

        # Reject all-zero inputs
        if (
            N == 0 or
            P == 0 or
            K == 0 or
            temperature == 0 or
            humidity == 0 or
            ph == 0 or
            rainfall == 0
        ):
            return render_template(
                'findyourcrop.html',
                prediction_text="Invalid input! Zelo values are not allowed.",
                values=values
            )

        # Validate realistic ranges
        if (
            N < 0 or N > 200 or
            P < 0 or P > 200 or
            K < 0 or K > 250 or
            temperature < -10 or temperature > 60 or
            humidity < 0 or humidity > 100 or
            ph < 0 or ph > 14 or
            rainfall < 0 or rainfall > 500
        ):
            return render_template(
                'findyourcrop.html',
                prediction_text="Invalid input! Please enter realistic agricultural values.",
                values=values
            )

        # Predict crop
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(data)

        return render_template(
            'findyourcrop.html',
            prediction_text=f"Recommended Crop: {prediction[0]}",
            values=values
        )

    except ValueError:
        return render_template(
            'findyourcrop.html',
            prediction_text="Please enter valid numeric values."
        )


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)