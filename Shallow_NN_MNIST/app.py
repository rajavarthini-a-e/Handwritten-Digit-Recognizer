import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="centered"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
model = load_model("mnist_shallow_model.keras")

# ==========================================
# TITLE
# ==========================================
st.title("✍️ Handwritten Digit Recognizer")

st.write(
    "Upload a handwritten digit image (0-9) and the trained Shallow Neural Network will predict it."
)

st.divider()

# ==========================================
# FILE UPLOADER
# ==========================================
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["png", "jpg", "jpeg"]
)

# ==========================================
# PROCESS IMAGE
# ==========================================
if uploaded_file is not None:

    st.subheader("Uploaded Image")

    st.image(uploaded_file, width=250)

    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize
    image = cv2.resize(image, (28, 28))

    # Invert colors
    image = 255 - image

    # Normalize
    image = image.astype("float32") / 255.0

    # Flatten
    image = image.reshape(1, 784)

    # Prediction
    prediction = model.predict(image)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.divider()

    st.success(f"Predicted Digit : {predicted_digit}")

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    st.balloons()