import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = os.path.join(base_dir, 'anemia_model.pkl')  # Update model file name
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details.html')
def details():
    return render_template('model.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Extracting the form data from the request
    Gender = float(request.form["Gender"])
    Age = float(request.form["Age"])
    Hemoglobin = float(request.form["Hemoglobin"])  # Relevant for anemia
    RedBloodCells = float(request.form["RedBloodCells"])  # Relevant for anemia
    IronLevel = float(request.form["IronLevel"])  # Relevant for anemia
    Symptoms = float(request.form["Symptoms"])  # General symptom input for anemia
    
    # Creating a numpy array of the input features
    features_values = np.array([[Gender, Age, Hemoglobin, RedBloodCells, IronLevel, Symptoms]])

    # Creating a DataFrame from the numpy array
    df = pd.DataFrame(features_values, columns=['Gender', 'Age', 'Hemoglobin', 'RedBloodCells', 'IronLevel', 'Symptoms'])

    # Making the prediction using the loaded model
    prediction = model.predict(df)
    print(prediction[0])

    # Interpreting the prediction result
    if prediction[0] == 0:
        result = "Mild Anemia"
        image_file = "mild_anemia.png"
    elif prediction[0] == 1:
        result = "Moderate Anemia"
        image_file = "moderate_anemia.png"
    elif prediction[0] == 2:
        result = "Severe Anemia"
        image_file = "severe_anemia.png"
    else:
        result = "Normal"
        image_file = "normal.png"

    print(result)

    # Preparing the response
    return render_template('result.html', result=result, image_file=image_file)


# if __name__ == "__main__":
#     app.run(debug=True)
