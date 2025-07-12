import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# loading in the model to predict on the data 
pickle_in = open('lg.pkl', 'rb') 
classifier = pickle.load(pickle_in)

def welcome(): 
    return 'Welcome all'

def prediction(sudden_fever,headache,mouth_bleed,nose_bleed,muscle_pain,joint_pain,
               vomiting,rash,swelling,chills,stomach_pain,orbital_pain,neck_pain,
               weakness,back_pain,weight_loss,red_eyes,slow_heart_rate,abdominal_pain,
               yellow_skin,yellow_eyes,breathing_restriction,itchiness):   
   
    prediction = classifier.predict( 
        [[sudden_fever,headache,mouth_bleed,nose_bleed,muscle_pain,joint_pain,
          vomiting,rash,swelling,chills,stomach_pain,orbital_pain,neck_pain,
          weakness,back_pain,weight_loss,red_eyes,slow_heart_rate,
          abdominal_pain,yellow_skin,yellow_eyes,breathing_restriction,itchiness]]) 
    print(prediction) 
    return prediction 

# Streamlit app
def main():

    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Vector Borne Disease Prediction</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.title('Disease Prediction App')

    # Define options
    options = ["Select", "No", "Yes"]

    # Collect user inputs
    sudden_fever = st.selectbox("Sudden Fever", options)
    headache = st.selectbox("Headache", options)
    mouth_bleed = st.selectbox("Mouth Bleed", options)
    nose_bleed = st.selectbox("Nose Bleed", options)
    muscle_pain = st.selectbox("Muscle Pain", options)
    joint_pain = st.selectbox("Joint Pain", options)
    vomiting = st.selectbox("Vomiting", options)
    rash = st.selectbox("Rash", options)
    swelling = st.selectbox("Swelling", options)
    chills = st.selectbox("Chills", options)
    stomach_pain = st.selectbox("Stomach Pain", options)
    orbital_pain = st.selectbox("Orbital Pain", options)
    neck_pain = st.selectbox("Neck Pain", options)
    weakness = st.selectbox("Weakness", options)
    back_pain = st.selectbox("Back Pain", options)
    weight_loss = st.selectbox("Weight Loss", options)
    red_eyes = st.selectbox("Red Eyes", options)
    slow_heart_rate = st.selectbox("Slow Heart Rate", options)
    abdominal_pain = st.selectbox("Abdominal Pain", options)
    yellow_skin = st.selectbox("Yellow Skin", options)
    yellow_eyes = st.selectbox("Yellow Eyes", options)
    breathing_restriction = st.selectbox("Breathing Restriction", options)
    itchiness = st.selectbox("Itchiness", options)

    # Convert 'Yes'/'No' to 1/0, and 'Select' to None
    input_values = [sudden_fever, headache, mouth_bleed, nose_bleed, muscle_pain, joint_pain, 
                    vomiting, rash, swelling, chills, stomach_pain, orbital_pain, neck_pain, 
                    weakness, back_pain, weight_loss, red_eyes, slow_heart_rate, abdominal_pain, 
                    yellow_skin, yellow_eyes, breathing_restriction, itchiness]

    # Check if all fields are selected
    if "Select" not in input_values:
        input_values = [1 if val == "Yes" else 0 for val in input_values]
        result = ""

        if st.button("Predict"): 
            result = prediction(*input_values)
            st.success(f'The output is {result}') 
    else:
        st.warning("Please fill in all the fields")

if __name__ == '__main__':
    main()
