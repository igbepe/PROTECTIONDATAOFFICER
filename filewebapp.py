#Description: This programe detects if someone has diabetes using machin learning and python
#import the Libreries

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create a tittle and a sub-title
st.write("""
    #Diabetes Detection Detect if someone has diabetes using machine learning and python !
""")

#Open and dispaly an image
image = Image.open('C:/Users/yolo/Documents/By_my_own/Web App/diabetes.jpg')
st.image(image, caption='ML', use_column_width=True)

#Get the Data
df = pd.read_csv('C:/Users/yolo/Documents/By_my_own/Web App/diabetes.csv')
#Set a subheader
st.subheader('Data Information:')

st.dataframe(df)
st.write(df.describe())
chart = st.bar_chart(df)
X = df.iloc[:, 0:8].values
Y = df.iloc[:,-1].values

#Split the data set into 75% Training and 25% Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0)

def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)
    #Store a dictonary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose':glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'insulin':insulin,
                 'BMI':BMI,
                 'DPD':DPF,
                 'age':age,
                 }
#Transforme data into a dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)
#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+ '%')

#Store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#Set a subheader and display classification
st.subheader('Classification: ')
st.write(prediction)
