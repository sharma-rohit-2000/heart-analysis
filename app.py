import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import 


st.write('Heart Attack Prediction analysis')


st.sidebar.header('Enter the correct value to get the perfect result')

st.sidebar.markdown(""" Add the feature values""")
dfa=pd.read_csv('heart.csv')

eatures=dfa.iloc[:,:-1]
target=dfa.iloc[:,-1]

uploaded_file = st.write(' ')

if uploaded_file is not None:
    pass
else:
    def main():
        age = st.sidebar.slider('Age',20,80)
        st.sidebar.write('0 stands for male')
        st.sidebar.write('1 stands for female')

        sex = st.sidebar.selectbox("Sex",(0,1))
        cp = st.sidebar.selectbox('chest pain',(0,1,2,3))
        trtbps = st.sidebar.slider('resting blood pressure (in mm Hg)',80,220)
        chol = st.sidebar.slider('cholestoral in mg/dl fetched via BMI sensor',100,600)
        fbs = st.sidebar.selectbox('(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)',(0,1))
        restecg = st.sidebar.selectbox('resting electrocardiographic results',(0,1,2))
        thalachh = st.sidebar.slider('maximum heart rate achieved',50,250)
        exng = st.sidebar.selectbox('exercise induced angina (1 = yes; 0 = no)',(0,1))
        oldpeak = st.sidebar.slider('Previous peak',1.0,7.0)
        slp = st.sidebar.selectbox('speech-language pathologist ',(0,1,2))
        caa = st.sidebar.selectbox('number of major vessel',(0,1,2,3,4))
        thal = st.sidebar.selectbox('Thalassemia ',(0,1,2,3))


        data={'Age':age,
          'Sex':sex,"CP":cp,'TrtBps':trtbps,'cholestrol':chol,'fbs':fbs,
          'Rest_Ecg':restecg,'Thalachh':thalachh,'exng':exng,'Old_Peak':oldpeak,
          'Slp':slp,'Caa':caa,'Thal':thal}
        features=pd.DataFrame(data,index=[1])
        return features
        st.write(features)
input_df = main()
st.write(input_df)
target=pd.read_csv('target.csv')


df=pd.concat([input_df,target],axis=1)

load_clf=pickle.load(open('randomforest.pkl','rb'))
prediction=load_clf.predict(input_df)
prediction_proba=load_clf.predict_proba(input_df)



st.subheader('Prediction')

result=np.array([0,1])

st.write(result[prediction])



st.subheader('Prediction Probabiltiy')
st.write(prediction_proba)
