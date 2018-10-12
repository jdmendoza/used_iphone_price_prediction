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
neigh = joblib.load("swappa_knn_model.pkl")

mlb_model = pickle.load(open('mlb_model.sav', 'rb'))
mlb_condition = pickle.load(open('mlb_condition.sav', 'rb'))

def predict_price(model,condition,size,coeff,intercept,knn_):
    test_vect = []

    if model == 'apple-iphone-7-plus-a1661':
        test_vect.extend(np.zeros(len(mlb_model.classes_)-1))
    else:
        model_list = mlb_model.transform([[model]])[0]
        print(model_list)
        test_vect.extend(np.delete(model_list,3))

    if condition == 'Mint':
        test_vect.extend(np.zeros(len(mlb_condition.classes_)-1))
    else:
        condition_list = mlb_condition.transform([[condition]])[0]
        test_vect.extend(np.delete(condition_list,2))

    test_vect.extend([size])

    knn_pred = ('$'+str(round(knn_.predict([test_vect])[0],2)))
    pred_price = np.dot(regr.coef_, np.transpose(test_vect))
    lasso_pred = ('$'+ str(round(pred_price+intercept,2)))

    return (lasso_pred,knn_pred)

'''End Prediction Stuff '''

class global_pred():
    pred_val_lasso = -1
    pred_val_knn = -1

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

        pred = predict_price(form.phone_model.data,form.condition.data,int(form.size.data),regr.coef_,regr.intercept_,neigh)
        global_pred.pred_val_lasso = pred[0]
        global_pred.pred_val_knn = pred[1]

        print('Yup', global_pred.pred_val_lasso, global_pred.pred_val_knn)

        return redirect(url_for("predict"))

    return render_template('home.html', form=form)

@app.route('/predict')
def predict():
    prediction = global_pred.pred_val_lasso
    prediction_knn = global_pred.pred_val_knn
    return render_template('predict.html', prediction=prediction, prediction_knn=prediction_knn)

if __name__ == "__main__":
    app.run()
