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
mlb_color = pickle.load(open('mlb_color.sav', 'rb'))
mlb_carrier = pickle.load(open('mlb_carrier.sav', 'rb'))
mlb_condition = pickle.load(open('mlb_condition.sav', 'rb'))

def predict_price(model,color,carrier,condition,size,coeff,intercept):
    test_vect = []

    base_price_new = {'apple-iphone-6s':649, 'apple-iphone-6s-plus':749, 'apple-iphone-7-a1660':649,
       'apple-iphone-7-plus-a1661':769, 'apple-iphone-8-a1863':699,
       'apple-iphone-8-plus-a1864':799, 'apple-iphone-se':399, 'apple-iphone-x-a1865':999}

    test_vect.extend(mlb_color.transform([[color]])[0])
    test_vect.extend(mlb_model.transform([[model]])[0])
    test_vect.extend(mlb_carrier.transform([[carrier]])[0])
    test_vect.extend(mlb_condition.transform([[condition]])[0])
    test_vect.extend([size])
    test_vect.extend([base_price_new[model]])
    print(len(test_vect))

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
    phone_model = StringField('What is the iPhone model?',validators=[DataRequired()])
    carrier  = StringField("Have carrier is it connected to?")
    color = StringField("What color is the phone?")
    size = StringField("How must memory is it? (in GB)")
    condition = StringField("What condition is it in?")
    submit = SubmitField('Submit')

@app.route('/', methods=['GET','POST'])
def index():
    form = InfoForm()

    if form.validate_on_submit():
        # Grab the data from the breed on the form.
        session['Phone Model'] = form.phone_model.data;
        #X_predict.append(int(form.phone_model.data))
        session['Carrier'] = form.carrier.data;
        #X_predict.append(int(form.carrier.data))
        session['Color'] = form.color.data;
        #X_predict.append(int(form.color.data))
        session['Condition'] = form.condition.data;
        #X_predict.append(float(form.condition.data))
        session['Size'] = form.size.data;
        global pred_val
        pred_val = predict_price(form.phone_model.data,form.color.data,form.carrier.data,form.condition.data,int(form.size.data),regr.coef_,regr.intercept_)
    	#X_predict.append(float(form.size.data))
        print('Yup', pred_val)
        return (redirect(url_for("predict")))
    return render_template('home.html', form=form)

@app.route('/predict')
def predict():
    prediction = pred_val
    return render_template('predict.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
