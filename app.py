import streamlit as st
import scipy.io
import numpy as np

# Basic helper for .mat model loading.
def load_mat_file(filename):
    data = scipy.io.loadmat(filename)
    st.write(data)  # show loaded data for debugging

    # IN PRODUCTION: parse the model variables
    # model = data.get('model_variable_name')
    # output = ... # add your SVM inference code here

# Page title
st.title("IoT Emotion Recognition Web App")

uploaded_file = st.file_uploader("Upload MATLAB .mat file", type=["mat"])
if uploaded_file is not None:
    # Save to disk then load with scipy.io
    with open("uploaded_mat_file.mat", "wb") as f:
        f.write(uploaded_file.getbuffer())
    load_mat_file("uploaded_mat_file.mat")


