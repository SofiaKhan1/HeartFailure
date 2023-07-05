from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('model22.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Heart Disease and Failure Prediction App"


@app.route('/predict', methods=['POST'])
def predict():
    Age = request.form.get('Age')
    Sex = request.form.get('Sex')
    ChestPainType = request.form.get('ChestPainType')
    RestingBP = request.form.get('RestingBP')
    Cholesterol = request.form.get('Cholesterol')
    FastingBS = request.form.get('FastingBS')
    RestingECG = request.form.get('RestingECG')
    MaxHR = request.form.get('MaxHR')
    ExerciseAngina = request.form.get('ExerciseAngina')
    Oldpeak = request.form.get('Oldpeak')
    ST_Slope = request.form.get('ST_Slope')

    input_query = np.array([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR,
                             ExerciseAngina, Oldpeak, ST_Slope]])

    result = model.predict(input_query)[0]

    return jsonify({'heart_disease': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
