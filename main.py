from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load((open('psychlogical_disorder.pkl', 'rb')))
app = Flask(__name__)


@app.route('/')
def home():
    return "hello world"


@app.route('/predict', methods=['POST'])
def predict():
    Symptom1 = request.form.get('Symptom1')
    Symptom2 = request.form.get('Symptom2')
    Symptom3 = request.form.get('Symptom3')
    Symptom4 = request.form.get('Symptom4')
    Symptom5 = request.form.get('Symptom5')

    input = np.array([[Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]])

    result = model.predict(input)[0]

    return jsonify({'Disorder': str(result)})


if __name__ == "__main__":
    app.run(debug=True)
