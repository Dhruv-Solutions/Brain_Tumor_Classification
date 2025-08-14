import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set page title
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image (JPG or PNG), and the model will predict the tumor type.")

# Load available models from the 'model1' folder
model_dir = 'Models1'
model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
model_options = {f.replace('.keras', ''): os.path.join(model_dir, f) for f in model_files}

if not model_options:
    st.error("No .keras models found in the 'model1' folder. Please add models.")
else:
    # Allow user to select a model
    selected_model_name = st.selectbox("Select a model to use:", list(model_options.keys()))
    model_path = model_options[selected_model_name]

    # Load the selected model
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Loaded model: {selected_model_name}")
    except Exception as e:
        st.error(f"Failed to load model {selected_model_name}: {str(e)}")
        st.stop()

# Define class names (update this list based on your actual classes)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


# File uploader for image input
uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))  # Adjust size based on model input
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display results
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")


# ---- 6. Footer ----
st.caption("Built with Streamlit and TensorFlow • MRI Brain Tumor Classifier")
