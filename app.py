import streamlit as st
import scipy.io
import numpy as np
from sklearn.svm import SVC
import joblib
import pandas as pd

st.title("IoT Emotion Recognition Web App")

uploaded_file = st.file_uploader("Upload MATLAB .mat file containing features and labels", type=["mat"])
if uploaded_file:
    # Save uploaded file locally
    with open("uploaded_mat_file.mat", "wb") as f:
        f.write(uploaded_file.getbuffer())

    matdata = scipy.io.loadmat("uploaded_mat_file.mat")
    st.subheader("Available Fields in .mat File")
    st.write(list(matdata.keys()))
    
    # Load features and labels (ensure your .mat includes both)
    if 'X' in matdata and 'y' in matdata:
        X = matdata['X']
        y = np.array([str(lbl[0]) for lbl in matdata['y']])  # Convert MATLAB cell array string labels
        
        st.write("Feature sample:")
        st.write(X[:5])
        st.write("Labels sample:")
        st.write(y[:5])

        # Train a model - this happens only once when file is uploaded
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X, y)

        # Save to disk for later use or you can directly predict here
        joblib.dump(clf, 'svm_emotion_model.pkl')
        st.success("Model trained and saved successfully!")

        # Predict on uploaded features itself (optional)
        predictions = clf.predict(X)
        st.write("Sample predictions on uploaded data:")
        st.write(predictions[:5])
    else:
        st.error("Uploaded .mat file does not contain both 'X' and 'y' variables.")

# Optional: If you want to upload test features separately for prediction
test_file = st.file_uploader("Upload test features CSV for prediction", type=["csv"])
if test_file:
    test_X = np.array(pd.read_csv(test_file, header=None))
    clf = None
    # Load your pretrained model if it exists
    try:
        clf = joblib.load('svm_emotion_model.pkl')
    except:
        st.error("Model not trained yet. Upload training .mat file first.")
    if clf:
        pred = clf.predict(test_X)
        st.write("Predictions for test features:")
        st.write(pred)

