
import cv2
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import pandas as pd
import time
import os
import statistics
import webbrowser
import detector as dt

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

st.image('imgs/streamlit-header_temp.png', use_column_width=True)

if st.button('Moodify me', key="center_button"):
    emotion = dt.detect_emotion()
    emotion_up = emotion.capitalize()
    st.write(f"The emotion is {emotion_up}!")
    
    subset = dt.subsets['sub_' + emotion]
    mood_result = subset.sample()
    link = mood_result['track_link'].iloc[0]
    webbrowser.open_new_tab(link)

    
    # Add the footer text at the bottom of the page
st.write(
    "<div style='position: fixed; bottom: 0; width: 100%; background-color: black; color: lightgrey; text-align: center; font-size:10px'>"
    "© 2023 Daniela Demarchi. All Rights Reserved."
    "</div>",
    unsafe_allow_html=True
)