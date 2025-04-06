import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model (.h5 file)
MODEL_PATH = "my_densenet_model.h5"  # Ensure the file is in the correct directory
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
classes = ["No Glaucoma", "Glaucoma"]

# Image preprocessing function
def preprocess_image(image):
    """Preprocess image for model prediction"""
    image = image.resize((224, 224))  # Resize to model's expected input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.set_page_config(page_title="Glaucoma Detection", layout="centered")
st.title(" Glaucoma Disease Detection ")

uploaded_file = st.file_uploader(" Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Debugging: Display raw prediction
    st.write(f"Raw Model Output: {prediction}")

    # Corrected prediction logic
    threshold = 0.5  # Adjust based on training performance
    predicted_class = classes[int(prediction[0, 0] > threshold)]

    # Display final result
    st.success(f"###  Diagnosis: {predicted_class}")
