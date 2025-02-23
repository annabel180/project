import streamlit as st
from tensorflow.keras.models import Sequential, model_from_json
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_model():
    return tf.keras.models.load_model("model.keras")  # Update with your model path

model = load_model()

# Define class labels (update based on your dataset)
class_labels = ['hen', 'horse', 'cow']  # Replace with actual class names

st.title("🐔🐴🐄 Animal Image Classifier")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels (RGB)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))  # Resize to match model input
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image_array)
    
    # Get the predicted class and confidence score
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Confidence score in %

    # Display prediction result
    st.subheader(f" Prediction: {predicted_class}")
    st.write(f" Confidence: **{confidence:.2f}%**")

    # Display class probabilities
    st.write(" **Class Probabilities:**")
    for i, label in enumerate(class_labels):
        st.write(f"- {label}: {prediction[0][i]*100:.2f}%")

  
