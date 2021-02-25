import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, url_for



app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('Home.html')
@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/prediction')
def prediction():
    return render_template('Predict.html')


@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Pregnancies','Glucose','BMI','Age']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        res_val = "Not Diabetic"
    else:
        res_val = "Diabetic"

    return render_template('Predict.html', prediction_text='Patient is {}'.format(res_val))

if __name__ == "__main__":
    app.run()
