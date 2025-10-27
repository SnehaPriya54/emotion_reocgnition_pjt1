import streamlit as st
import scipy.io
import numpy as np
from sklearn.svm import SVC
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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

        # Balance check
        unique, counts = np.unique(y, return_counts=True)
        st.write("Label distribution:", dict(zip(unique, counts)))

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Grid search for best SVM (tune C and gamma for RBF)
        param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['rbf']}
        grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=5)
        grid.fit(X_scaled, y)
        clf = grid.best_estimator_
        st.success(f"Model trained and saved successfully! Best params: {grid.best_params_}")

        # Save both scaler and model for future use
        joblib.dump({'scaler': scaler, 'model': clf}, 'svm_emotion_model.pkl')

        # Predict on uploaded features itself (optional)
        predictions = clf.predict(X_scaled)
        st.write("Sample predictions on uploaded data:")
        st.write(predictions[:5])
    else:
        st.error("Uploaded .mat file does not contain both 'X' and 'y' variables.")

# Optional: If you want to upload test features separately for prediction
test_file = st.file_uploader("Upload test features CSV for prediction", type=["csv"])
if test_file:
    test_X = np.array(pd.read_csv(test_file, header=None))
    try:
        data = joblib.load('svm_emotion_model.pkl')
        scaler = data['scaler']
        clf = data['model']
        test_X_scaled = scaler.transform(test_X)
        pred = clf.predict(test_X_scaled)
        st.write("Predictions for test features:")
        st.write(pred)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
