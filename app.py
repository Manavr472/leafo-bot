import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load the trained model
model_path = './leaf_classification_model_152v2.keras'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error(f"Model file not found at {model_path}")

# Load the class names
class_names_path = 'class_names.npy'
if os.path.exists(class_names_path):
    class_names = np.load(class_names_path, allow_pickle=True).item()
else:
    st.error(f"Class names file not found at {class_names_path}")

def predict_img(path):
    img = image.load_img(path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr_expnd = np.expand_dims(img_arr, axis=0)
    img = tf.keras.applications.resnet_v2.preprocess_input(img_arr_expnd)

    pred = model.predict(img)
    result = class_names[np.argmax(pred)]

    return result

st.title("Leaf Classification App")

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Save the uploaded image to a temporary file
    temp_file_path = "temp_image.png"
    image.save(temp_file_path)
    
    # Make prediction
    result = predict_img(temp_file_path)

    # Display the prediction
    st.write(f"Predicted class: {result}")

# Optionally, display some information about the classes
st.write("This model can classify 170 different types of leaves.")
