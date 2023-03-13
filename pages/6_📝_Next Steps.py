import streamlit as st
import pandas as pd
import plotly.express as px

from matplotlib import pyplot as plt

with open('style.css') as f: 
    st.markdown(f'<style>{f.read()} </style>', unsafe_allow_html=True)

st.markdown( """ <style> .sidebar .sidebar-content {color: black} </style> """, unsafe_allow_html=True, )

container_style = """
    ..sidebar .sidebar-content {
        background-color: #212121;
        border: 1px solid #FFFFFF;
        border-radius: 10px;
        padding: 20px;
        color: #D3D3D3;
        font-size: 1.1rem;
        font-family: 'Open Sans', sans-serif;
        
    } """