import streamlit as st
from PIL import Image
import base64

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

st.markdown("<h1 style='color:#9FE1B4; font-family: serif;'> Moodify: the app that matches a song to your mood</h1>", unsafe_allow_html=True)
#st.markdown("<h3 style='color:#9FE1B4; font-family: serif;'> Meet Moodify, the app that matches your jam to your mood</h3>", unsafe_allow_html=True)
file_ = open("imgs/moodify_demo_gif_2.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="model-gif" width="695" height="385"><br> <br>',
    unsafe_allow_html=True
)
st.write(
    "<div style='font-size:1.2rem'>"
    "Moodify is an application that leverages facial recognition technology to personalize music recommendations on Spotify. <br><br> By analysing the user\'s facial expressions in real-time using the device\'s camera, Moodify can infer the user\'s current mood and suggest songs that match their emotional state. This project explores the opportunity we have to harness the power of computer vision to create more meaningful and personalized interactions with technology.\n My aim was to tackle a more the human aspect of data analytics by going beyond the quantitative analysis of data and incorporating the qualitative aspects of human experience, such as perception, cognition and emotion. Computer vision, enables machines to interpret visual data in ways that resembles human perception, such as recognizing objects, scenes and facial expressions. <br><br> <b> By incorporating these human aspects into data analytics we can gain deeper insights into complex phenomena such as social interactions cultural norms and psychological states, that are not easily captured by numerical data alone.</b>\
    <div style='text-align: center; font-size: 1.5rem; font-weight: bold;'><br><br>But how can we teach a computer to recognize emotions?</div>\
    </div>",
    unsafe_allow_html=True
)
