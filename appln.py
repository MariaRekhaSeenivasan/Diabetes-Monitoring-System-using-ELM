from flask import Flask, request, render_template, flash
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecret'

# Load the scaler and PCA model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

# Load the ELM model
with open('elm_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        pregs = int(request.form.get('pregs'))
        gluc = int(request.form.get('gluc'))
        bp = int(request.form.get('bp'))
        skin = int(request.form.get('skin'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        func = float(request.form.get('func'))
        age = int(request.form.get('age'))

        # Preprocess the input data
        input_data = pd.DataFrame([[pregs, gluc, bp, skin, insulin, bmi, func, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Include the outcome column from the original data
        #input_data['Outcome'] = 0  # Replace 0 with the appropriate value if available

        # Apply the scaler and PCA transformation
        input_scaled = scaler.transform(input_data)
        input_pca = pca.transform(input_scaled)

        # Predict using the model
        prediction = model.predict(input_pca)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
