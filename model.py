import numpy as np
import pandas as pd
import pickle
import os

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = os.path.join(base_dir, 'anemia_model.pkl')  # Ensure your model file is named correctly
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Custom input features (for example)
custom_input = {
    'Gender': 1,             # 1 for Male, 0 for Female
    'Age': 25,               # Age in years
    'Hemoglobin': 13.5,      # Hemoglobin level in g/dL
    'RedBloodCells': 5.2,    # RBC count in million/µL
    'IronLevel': 80,          # Iron level in µg/dL
    'Symptoms': 0            # Severity of symptoms (0 for None)
}

# Creating a numpy array of the input features
features_values = np.array([[custom_input['Gender'], custom_input['Age'], custom_input['Hemoglobin'],
                             custom_input['RedBloodCells'], custom_input['IronLevel'],
                             custom_input['Symptoms']]])

# Creating a DataFrame from the numpy array
df = pd.DataFrame(features_values, columns=['Gender', 'Age', 'Hemoglobin', 'RedBloodCells', 'IronLevel', 'Symptoms'])

# Making the prediction using the loaded model
prediction = model.predict(df)
print(f"Predicted Anemia Stage: {prediction[0]}")

# Interpreting the prediction result
if prediction[0] == 0:
    result = "Mild Anemia"
elif prediction[0] == 1:
    result = "Moderate Anemia"
elif prediction[0] == 2:
    result = "Severe Anemia"
else:
    result = "Normal"

print(f"Your predicted condition is: {result}")
