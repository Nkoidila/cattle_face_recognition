import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Path to the dataset
data_path = r'C:\Users\kores\OneDrive\Desktop\Ngishu\cattle_set2'


# Initializing label map
label_map = {}
current_label = 0

# Create the label map
for label_folder in os.listdir(data_path):
    label_folder_path = os.path.join(data_path, label_folder)
    if os.path.isdir(label_folder_path):
        label_map[current_label] = label_folder
        current_label += 1

# Load the model
model_path = r'C:\Users\kores\OneDrive\Desktop\Ngishu\cattle_model5.keras'
model = load_model(model_path)

#function to preprocess the image
def preprocess_image(image_path, img_height=128, img_width=128):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

#function for predicting the cow name
def predict_cow(image_path, model, label_map):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction, axis=1)[0]
    cow_identity = label_map[predicted_label]
    return cow_identity

# Streamlit app
st.title("Cattle Face Recognition")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the image
    img = preprocess_image("temp.jpg")

    # Predict the cow identity
    cow_identity = predict_cow("temp.jpg", model, label_map)

    # Display the image and prediction
    st.image("temp.jpg", caption='Uploaded Image.', use_column_width=True)
    st.write(f"The cow in the image is: {cow_identity}")
