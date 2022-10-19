# flask, scikit-learn,pandas,picle-mixin
from email.quoprimime import body_check
import pandas as pd
from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)
data = pd.read_csv('Cleaned_House_data.csv')
pipe=pickle.load(open("LR_Model.pkl",'rb'))


@app.route('/')
def index():

    locations=sorted(data['Location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    sqft=request.form.get('total_sqft')

    print(location,bhk,sqft)

    input = pd.DataFrame([[sqft,location,bhk]],columns=['Area','Location','No. of Bedrooms'])
    prediction = pipe.predict(input)[0]
    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True)

