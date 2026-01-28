from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('temperature_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from JSON data
        year = int(data['year'])
        month = int(data['month'])
        district = data['district']
        parameter = data['parameter']
        
        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            'year': [year],
            'month': [month],
            'district_name': [district],
            'parameter': [parameter]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': round(prediction, 2),
            'units': '°C'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)