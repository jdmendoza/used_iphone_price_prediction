from flask import Flask, render_template, session, redirect, url_for, session
import numpy as np
from flask_wtf import FlaskForm
from wtforms import (StringField, RadioField, DecimalField, SubmitField)
from wtforms.validators import DataRequired
import pickle

''''
For Prediction
'''
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib


regr = joblib.load("swappa_lr_model.pkl")

mlb_model = pickle.load(open('mlb_model.sav', 'rb'))
mlb_condition = pickle.load(open('mlb_condition.sav', 'rb'))

def predict_price(model,condition,size,coeff,intercept):
    test_vect = []

    if model == 'apple-iphone-7-plus-a1661':
        test_vect.extend(np.zeros(len(mlb_model.classes_)-1))
    else:
        model_list = mlb_model.transform([[model]])[0]
        print(model_list)
        test_vect.extend(np.delete(model_list,3))
        #test_vect.insert(3,0)
    print(len(test_vect))

    if condition == 'Mint':
        test_vect.extend(np.zeros(len(mlb_condition.classes_)-1))
    else:
        condition_list = mlb_condition.transform([[condition]])[0]
        test_vect.extend(np.delete(condition_list,2))

    test_vect.extend([size])

    pred_price = np.dot(regr.coef_, np.transpose(test_vect))

    return '$'+ str(round(pred_price+intercept,2))
'''End Prediction Stuff '''

#from file import predict
#from file import transform

#loaded_model = pickle.load(open('model.sav', 'rb'))
pred_val = -1
app=Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

class InfoForm(FlaskForm):
    '''
    This general class gets a lot of form about a person on the Titanic.
    Mainly a way to go through many of the WTForms Fields.
    '''
    phone_model = RadioField('Choose your model?', choices=[('apple-iphone-6s','iPhone 6s'),('apple-iphone-6s-plus','iPhone 6s Plus'),('apple-iphone-7-a1660','iPhone 7'),
    ('apple-iphone-7-plus-a1661','iPhone 7 Plus'),('pple-iphone-8-a1863','iPhone 8'),('apple-iphone-8-plus-a1864','iPhone 8 Plus'),
    ('apple-iphone-se','iPhone SE'),('apple-iphone-x-a1865','iPhone X')])

    size = StringField("How must memory is it? (integer)")
    condition = RadioField("What condition is it in?",choices=[('New (Resale)','New'),('Mint','Mint'),('Good','Good'),('Fair','Fair')])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET','POST'])
def index():
    form = InfoForm()

    if form.validate_on_submit():

        # Grab the data from the breed on the form.
        session['Phone Model'] = form.phone_model.data;
        session['Condition'] = form.condition.data;
        session['Size'] = form.size.data;
        global pred_val
        pred_val = predict_price(form.phone_model.data,form.condition.data,int(form.size.data),regr.coef_,regr.intercept_)
        print('Yup', pred_val)

        return (redirect(url_for("predict")))

    return render_template('home.html', form=form)

@app.route('/predict')
def predict():
    prediction = pred_val
    return render_template('predict.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
