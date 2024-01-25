# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 13:49:30 2024

@author: DELL
"""
## Streamlit Application for Sentiment Analysis

import streamlit as st
import pickle
from PIL import Image
import base64
with open("C:\\Users\\DELL\\Downloads\\Sentiment Analysis\\Log_Model_S.pkl",'rb') as f:
    model=pickle.load(f)
with open("C:\\Users\\DELL\\Downloads\\Sentiment Analysis\\vec.pkl",'rb') as g:
    vec=pickle.load(g)


st.title("Sentiment Analyzer")
user_input=st.text_area("Enter the text")

## Setting up of background Image
if user_input:
    def add_bg_from_local(image_file):
        
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        
    add_bg_from_local('C:/Users/DELL/Downloads/SA_Img.png')
    # Transforming the given sentence to Vectors
    user_input_preprocess=vec.transform([user_input]) 
    # Predicting the Model
    prediction=model.predict(user_input_preprocess)[0]
    sentiment_label = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}[prediction]

    # Display the prediction
    if st.button('Submit'):
        import time
        with st.spinner("wait for it"):
            time.sleep(2)
    st.markdown(f"<p style='color:Red; font-style: italic; font-weight: bold;'>Predicted Sentiment: {sentiment_label}</p>", unsafe_allow_html=True)