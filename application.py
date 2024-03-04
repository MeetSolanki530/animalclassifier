import streamlit as st
import os
# import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('vgg16_animal_classifier_model_with_early_stopping_1.h5')

# Define the categories
categories = ['Domestic', 'Not Domestic']

def preprocess_image(image):
    # Resize and preprocess the image
    image = cv2.resize(image, (150, 150))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_domestic(image):
    # Preprocess the image
    image = preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(image)
    
    # Interpret predictions
    probability_domestic = predictions[0][0]  # Probability of being domestic
    probability_not_domestic = predictions[0][1]  # Probability of not being domestic
    
    return probability_domestic, probability_not_domestic

# Streamlit app
def main():
    st.title('Animal Domesticity Classifier')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        probability_domestic, probability_not_domestic = predict_domestic(image)
        st.write(f"There is {probability_domestic * 100:.2f}%, Probability that animal is Domestic.")
        st.write(f"There is {probability_not_domestic * 100:.2f}%, Probability that animal is Wild.")

if __name__ == "__main__":
    main()

