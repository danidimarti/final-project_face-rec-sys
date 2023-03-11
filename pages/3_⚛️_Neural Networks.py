import streamlit as st


st.title('What is a Neural Network?')
st.subheader('A multi-layered learning model')
# Load the GIF using the open() function
with open("imgs/neural_networks_gif.gif", "rb") as f:
    gif_bytes = f.read()

# Display the GIF using st.image()
st.image(gif_bytes)