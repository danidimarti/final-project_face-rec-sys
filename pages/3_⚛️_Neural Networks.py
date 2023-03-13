import streamlit as st

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

    
st.markdown("<h1 style='color:#9FE1B4; font-family: serif;'>What is a Neural Network?</h1>", unsafe_allow_html=True)
st.subheader('A multi-layered learning model')
# Load the GIF using the open() function
with open("imgs/neural_networks_gif.gif", "rb") as f:
    gif_bytes = f.read()

# Display the GIF using st.image()
st.image(gif_bytes)