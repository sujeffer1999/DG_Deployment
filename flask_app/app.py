import numpy as np
import pickle
from flask import Flask, request, render_template, url_for

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = request.form['sl']
    sepal_width = request.form['sw']
    petal_width = request.form['pw']

    prediction = model.predict(np.asarray([sepal_length, sepal_width, petal_width], dtype=float).reshape(-1,3))

    output = np.round(prediction[0, 0], 2)

    return render_template('index.html', prediction_text='Predicted Petal Length is {} with sepal length: {}, sepal'
                                                         ' width: {} and petal width: {}'.format(output,
                                                                                                 sepal_length,
                                                                                                 sepal_width,
                                                                                                 petal_width))
