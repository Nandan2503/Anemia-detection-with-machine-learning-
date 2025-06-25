import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = os.path.join(base_dir, 'model.pkl')
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details.html')
def details():
    return render_template('details.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # Extracting the form data from the request
    Gender = float(request.form["Gender"])
    Age = float(request.form["Age"])
    Hemoglobin = float(request.form["Hemoglobin"])
    MCV = float(request.form["MCV"])
    MCH = float(request.form["MCH"])
    RBC = float(request.form["RBC"])

    # Add more features if required by your model

    features_values = np.array([[Gender, Age, Hemoglobin, MCV, MCH, RBC]])
    df = pd.DataFrame(features_values, columns=['Gender', 'Age', 'Hemoglobin', 'MCV', 'MCH', 'RBC'])

    prediction = model.predict(df)

    if prediction[0] == 0:
        result = "No Anemia"
        image_file = "normal.png"
    elif prediction[0] == 1:
        result = "Mild Anemia"
        image_file = "mild.png"
    else:
        result = "Severe Anemia"
        image_file = "severe.png"

    return render_template('result.html', result=result, image_file=image_file)

# To run the app:
# if __name__ == "__main__":
#     app.run(debug=True)
