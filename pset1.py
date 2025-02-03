import streamlit as st
from fastai.vision.all import *

import os
import pathlib
from pathlib import Path
import sys

# Dynamically patch pathlib for cross-platform compatibility
if sys.platform == "win32":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath

# Load the trained model
def load_model():
    try:
        model_path = os.path.abspath('mongolian_foods_classifier_unix.pkl')
        model = load_learner(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    finally:
        # Restore the original pathlib settings
        if sys.platform == "win32":
            pathlib.PosixPath = temp
        else:
            pathlib.WindowsPath = temp

# Streamlit app
def main():
    st.title("Mongolian Foods Classifier")
    st.write("Upload an image of Mongolian food (buuz, khuushuur, tsuivan, or bansh) to classify it.")

    # Load the model
    model = load_model()
    if model is None:
        return

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Make a prediction
        if st.button("Classify"):
            img = PILImage.create(uploaded_file)
            pred, pred_idx, probs = model.predict(img)
            st.write(f"**Prediction**: {pred}")
            st.write(f"**Probability**: {probs[pred_idx]*100:.02f}%")

# Run the app
if __name__ == "__main__":
    main()
