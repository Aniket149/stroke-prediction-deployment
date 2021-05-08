import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        prediction = 'You Have Stroke,Visit Doctor'
    else:
        prediction = "You Don't Have Stroke"

    return render_template('index.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)