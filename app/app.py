from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load(f'app/model/insurance_charges_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    features = [int(data['age']), int(data['sex']), float(data['bmi']), int(data['children']), int(data['smoker']),
                int(data['region_northwest']), int(data['region_southeast']), int(data['region_southwest'])]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction_text='Predicted Insurance Charge: ${:.2f}'.format(prediction))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
