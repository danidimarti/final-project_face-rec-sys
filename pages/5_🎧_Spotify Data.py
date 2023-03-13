import streamlit as st
import pandas as pd
import plotly.express as px

from matplotlib import pyplot as plt

# Define the CSS style for the container
container_style = """
    .text-container {
        background-color: #212121;
        border: 1px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.1rem;
        font-family: 'Open Sans', sans-serif;
        
    }
"""

bullet_style = """
    .bullet {
        background-color: #212121;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.3rem;
        font-family: 'Open Sans', sans-serif;
        
    }
"""

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

    
st.markdown("<h1 style='color:#9FE1B4; font-family: serif;'>Spotify Data</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='color:#9FE1B4;'>Emotion labeled dataset from Spotify Api</h3>", unsafe_allow_html=True)

st.write(
    "<div style='font-size:1.2rem'>"
    "The Spotify API provides developers with access to a massive library of songs and metadata. I utilized this API to extract songs from playlists that corresponded to the four emotion categories that I selected when creating the CNN model: angry, happy, sad, and neutral.</div>",
    unsafe_allow_html=True
)

######### ------ SPOTIFY SETTINGS  ------------- ########

token = "<div style='font-size:1.2rem;'><ol type='1'><li>Get auth for App development (CLIENT_ID, CLIENT_SECRET) </li><li>Get auth for token for the queries.</li></ol></div>"

st.markdown(f'<div class="bullet-points"><style>{bullet_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.7rem">Spotify Token</div>{token}</div><br>', unsafe_allow_html=True)


######----- GET MOOD PLAYLISTS ------ ######

getplaylists = """
<div style='font-size:1.2rem;'>
<ol>
  <li>Find mood playlists for each of the Model sentiments used</li>
  <li>Extract playlist id</li>
  <li>Create function to extract playlist data from Api based on the structure of the response dictionary.</li>
  <li>Create emotion column for each dictionary</li>
  <li>Append information to list</li>
  <li>Create df</li>
  <li>Save csv file</li>
</ol>
</div>
"""

st.markdown(f'<div class="bullet-points"><style>{bullet_style}</style><div style="color:#FFDAB9; font-family: serif; font-size:1.7rem">Steps:</div>{getplaylists}</div><br>', unsafe_allow_html=True)


##### ----- VISUALIZE THE DF ------ ####
st.write(
    "<div style='font-size:1.2rem;'>"
    "I used a total of 15 playlists, from the app's mood feature. This feature allows users to browser playlists and songs that are curated based  on different emotions. The resulting df looks like this:</div><br>",
    unsafe_allow_html=True
)

st.image('imgs/Spotify_sample.png', caption='Fer2013_Emotion Count', use_column_width=True)

st.write(
    "<div style='font-size:1.2rem;'>"
    "<br>Let\'s visualize the number of songs per emotion we have:</div><br>",
    unsafe_allow_html=True
)

st.image('imgs/Spotify_label-count.png', caption='Fer2013_Emotion Count', use_column_width=True)



