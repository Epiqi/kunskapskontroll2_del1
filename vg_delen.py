import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image

# 🔹 Ladda den tränade modellen
model = joblib.load("best_mnist_model.joblib")

# 🔹 Funktion för att förbereda bilden
def preprocess_image(image):
    image = image.convert('L')  # Gråskala
    image = image.resize((28, 28))  # Skala till 28x28
    image = np.array(image)  # Konvertera till array
    image = cv2.bitwise_not(image)  # Invertera färger (vitt->svart, svart->vitt)
    image = image / 255.0  # Normalisera
    image = image.flatten().reshape(1, -1)  # Flatten till en 1D-vektor (784 pixlar)
    return image

# 🔹 Streamlit-gränssnitt
st.title("🖌️ MNIST Digit Recognition")
st.write("Rita en siffra i rutan nedan och tryck på **'Prediktera'**!")

# Rita på canvas
canvas = st.canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 🔹 Prediktion
if st.button("Prediktera"):
    if canvas.image_data is not None:
        image = Image.fromarray((canvas.image_data[:, :, :3] * 255).astype(np.uint8))  # Ta bort alfa-kanalen
        processed_image = preprocess_image(image)
        
        # Gör prediktion
        prediction = model.predict(processed_image)
        st.write(f"🧠 **Predikterad siffra:** {prediction[0]}")
    else:
        st.write("⚠️ Rita en siffra först!")

