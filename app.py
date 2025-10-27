import streamlit as st
import scipy.io
import numpy as np

from sklearn.svm import SVC
import joblib

st.title("IoT Emotion Recognition Web App")

uploaded_file = st.file_uploader("Upload MATLAB .mat file", type=["mat"])
if uploaded_file:
    st.success("File Uploaded Successfully")
    with open("uploaded_mat_file.mat", "wb") as f:
        f.write(uploaded_file.getbuffer())

    matdata = scipy.io.loadmat("uploaded_mat_file.mat")

    # Step 1: List all top-level keys
    st.subheader("Available Fields in .mat File")
    st.write(list(matdata.keys()))

    # Step 2: Try to get compactStruct, the most likely SVM container
    compact_struct = matdata.get('compactStruct', None)
    if compact_struct is not None:
        st.subheader("Sample of 'compactStruct' Contents")
        st.write(compact_struct)
        st.info(
            "Direct inference with this MATLAB SVM model is not possible in Python.\n"
            "To perform predictions in Streamlit, you will need to export features/labels from MATLAB and retrain an SVM in Python (e.g., scikit-learn), save as .pkl, and load that for prediction."
        )
    else:
        st.warning("No 'compactStruct' found. Top-level fields: " + str(list(matdata.keys())))

    # Step 3: If you have features/labels, visualize them
    # Example: 
    # if 'features' in matdata and 'labels' in matdata:
    #     features = matdata['features']
    #     labels = matdata['labels']
    #     st.write("Feature shape:", features.shape)
    #     st.write("Label shape:", labels.shape)
    #     st.write("First 5 rows:", features[:5], labels[:5])

    # --- For actual ML prediction, see note below ---

st.markdown("""
**Next steps:**
- For true prediction, upload CSV/test features and load a scikit-learn model trained in Python.
- Use the keys above to explore and map what is saved in your .mat file for further migration.
""")





data = scipy.io.loadmat('emotion_features.mat')
X = data['X']
y = np.array([str(lbl[0]) for lbl in data['y']])  # if 'y' is saved as a cell array of strings

clf = SVC(kernel='rbf', probability=True)
clf.fit(X, y)
joblib.dump(clf, 'svm_emotion_model.pkl')


st.write(type(compact_struct))
st.write("Model loaded, but not displayed.")
