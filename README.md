# diabetes
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 23:26:42 2024

@author: LENOVO
"""

import numpy as np
import streamlit as st
import joblib
from PIL import Image
d=joblib.load("C:/Users/LENOVO/Downloads/archive (3)/diabetes_ai.joblib")


def diabetes(input_data):
    input_data=(166,72,19,175,25.8,0.587)
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
  
    result=d.predict(input_data_reshaped)
    print("___________SEE THE RESULT__________")
    if result>0.5:
        return "Positive"
    else:
        return "no Negative"

def main():
    st.title("diabetes prediction")

    
    Glucose=st.text_input("number of Glucose")
    BloodPressure=st.text_input("number of BloodPressure")
    SkinThickness=st.text_input("number of SkinThickness")
    Insulin=st.text_input("number of Insulin")
    BMI=st.text_input("number of BMI")
    DiabetesPedigreeFunction=st.text_input("number of DiabetesPedigreeFunction")
    
    
    
    diagnosis=''
    
    if st.button("diabetes test result"):
        diagnosis=diabetes([Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction])
        st.success(diagnosis)


if __name__ == '__main__':
    main()
    
        
