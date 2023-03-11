import streamlit as st
from PIL import Image

st.title('About this project')
st.subheader('Meet Moodify, the app that matches your jam to your mood')

st.write(
    "<div style='font-size:1rem'>"
    "Moodify is an application that leverages facial emotion recognition technology to personalize music recommendations on Spotify. By analysing the user\'s facial expressions in real-time using the device\'s camera, Moodify can infer the user\ 's current mood and suggest songs that match emotional state. This project explores the opportunity we have to harness the power of computer vision to create more meaningful and personalized interactions with technology.\
    \n\nMy aim was to tackle a more the human aspect of data analytics by going beyond the quantitative analysis of data and incorporating the qualitative aspects of human experience, such as perception, cognition and, in this case, emotion. Computer vision, enables machines to interpret visual data in ways that resembles human perception, such as recognizing objects, scenes and facial expressions. <b>  *By incorporating these human aspects into data analytics we can gain deeper insights into complex phenomena such as social interactions cultural norms and psychological states, that are not easily captured by numerical data alone*.</b>\
    <div style='text-align: center; font-size: 1.2rem; font-weight: bold;'>But how can we teach a computer how to recognize emotions?</div>\
    </div>",
    unsafe_allow_html=True
)
