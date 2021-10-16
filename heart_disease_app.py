# -*- coding: utf-8 -*-
"""
Created on Sat May  1 10:08:01 2021

@author: Lenovo
"""

import streamlit as st
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
import matplotlib.image as mpimg
import pickle


st.set_page_config(page_title='Heart Disease Detection Machine Learning App',layout='wide')
imageha = mpimg.imread('ha.jpg')     

st.write("""
# Heart Disease Detection Machine Learning App

In this implementation, various **Machine Learning** algorithms are used in this app for building a **Classification Model** to **Detect Heart Disease**.
""")
st.image(imageha)

data = pd.read_csv('heart.csv')
X = data[['age', 'sex', 'cp', 'trestbps', 'chol',
       'fbs', 'restecg', 'thalach','exang','oldpeak','slope','ca','thal']]
y = data['target']


Selection = st.sidebar.selectbox("Select Option", ("Heart Disease Detection App","Exploratory Data Analysis","Source Code"))

if Selection == "Heart Disease Detection App":
    
    #st.markdown('The Diabetes dataset used for training the model is:')
    #st.write(data.head(5))
    
    st.write("'cp' - chest pain")
    st.write("'testbps' - resting blood pressure (in mm Hg on admission in hospital)")
    st.write("'chol' - serum cholestrol in mg/dl")
    st.write("'fbs' - (fasting blood sugar > 120mg/dl) 1 = true, 0 = false")
    st.write("'restecg'- Rest ECG")
    st.write("'exang' - exercise induced angina")
    st.write("'oldpeak' - ST depression induced by exercise related to rest")
    st.write("'slope' - the slope of the peak exercise ST segment")
    st.write("'ca' - number of major vessels(0-3) colored by flourosopy")
    st.write("'thal' - (0-3) 3 = normal; 6 = fixed defect; 7 = reversable defect")
    st.write("'target' - 1 or 0")
    
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(data.head(5))
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
     #st.sidebar.header('2. Set Parameters'):
    age = st.sidebar.slider('age',29,77,40,1)
    cp = st.sidebar.slider('cp', 0, 3, 1, 1)
    sex = st.sidebar.slider('sex',0,1,0,1)
    trestbps = st.sidebar.slider('trestbps', 94, 200, 80, 1)
    chol = st.sidebar.slider('chol', 126, 564, 246, 2)
    fbs = st.sidebar.slider('fbs', 0, 1, 0, 1)
    restecg = st.sidebar.slider('restecg', 0, 2, 1, 1)
    exang = st.sidebar.slider('exang', 0, 1, 0, 1)
    oldpeak = st.sidebar.slider('oldpeak', 0.0, 6.2, 3.2, 0.2)
    slope= st.sidebar.slider('slope', 0, 2, 1, 1)
    ca= st.sidebar.slider('ca', 0, 4, 2, 1)
    thal= st.sidebar.slider('thal', 0, 3, 1, 1)
    thalach = st.sidebar.slider('thalach',71,202,150,1)
    
    X_test_sc = [[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]

    #logregs = LogisticRegression()
    #logregs.fit(X_train, y_train)
    #y_pred_st = logregs.predict(X_test_sc)
    
    load_clf = pickle.load(open('heart_clf.pkl', 'rb'))

# Apply model to make predictions
    prediction = load_clf.predict(X_test_sc)
    prediction_proba = load_clf.predict_proba(X_test_sc)
    
    answer = prediction[0]
        
    if answer == 0:

        st.title("**The prediction is that the Heart Disease was not Detected**")
   
    else:   
        st.title("**The prediction is that the Heart Disease was Detected**")
        
    st.write('Note: This prediction is based on the Machine Learning Algorithm, Logistic Regression.')
    


