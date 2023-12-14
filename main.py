import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import os

from utils import preprocess_image

animals = ['cat', 'dog', 'wild']

def load_model():
    loc = os.path.join(os.getcwd(), 'models', 'model.h5')
    localhost_load_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    model = tf.keras.models.load_model(loc, options=localhost_load_option)
    return model

model = load_model()

st.title('Animal Faces 🐶🐱🦊')

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.write("Classifying...")    

        # Resize
        processed_image = preprocess_image(image, target_size=(256, 256))  # Modify this line based on your model

        # Predict the class
        prediction = model.predict(processed_image)

        scores = tf.nn.softmax(prediction[0])  # Assuming a softmax final layer

        # Get the highest probability class
        predicted_class = np.argmax(scores, axis=0)
        probability = np.max(scores)

        st.image(image, use_column_width=True)
        st.header(f'Prediction: {animals[predicted_class]}')
