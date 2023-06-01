# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 19:15:48 2022

@author: HP
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    # changing the input data to numpy array
    input_data_as_numpy_array = np.array(input_data)
    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0] == 0):
        return "Person is not diabetic"
    else:
        return "Person is diabetic"


def main():
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('BloodPressure Value')
    SkinThickness = st.text_input('SkinThickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
    Age = st.text_input('Age of the Person')
    
    
    # code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)


if __name__ == '__main__':
    main()
    
        
        
    


        