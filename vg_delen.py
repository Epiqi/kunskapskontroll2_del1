import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas  # Korrekt import
from sklearn.preprocessing import StandardScaler


# Ladda den tränade modellen
model = joblib.load("best_mnist_model.joblib")
scaler = joblib.load("scaler.joblib") 

st.title('MNIST Digit Recognizer')
st.markdown('''
Skriv en siffra här!
''')


SIZE = 192
canvas_result = st_canvas(
    fill_color='#FFFFFF',
    stroke_width=20,
    stroke_color='#000000',
    background_color='#FFFFFF',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')



if st.button('Förutspå'):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
        img = img.convert('L')
        img = img.resize((28, 28), Image.BILINEAR)

        img_as_array = np.array(img)
        img_as_array = 255 - img_as_array

        img_as_array = img_as_array.flatten().reshape(1, -1)
        img_as_array_scaled = scaler.transform(img_as_array)

        st.write(f'förbehandlad bild:')
        st.image(img, width=100)

        prediction = model.predict(img_as_array_scaled)
        st.write(f'Prediktion: {prediction[0]}')
    else:
        st.write('Ingen bild finns att förutspå')