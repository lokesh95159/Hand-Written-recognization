import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from textblob import TextBlob

# -----------------------------
# Load EasyOCR
# -----------------------------
@st.cache_resource
def load_ocr():
    reader = easyocr.Reader(['en'])
    return reader

reader = load_ocr()

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # OCR
    results = reader.readtext(image_np)

    raw_text = ""
    for (_, text, prob) in results:
        raw_text += text + " "

    st.subheader("Raw Output:")
    st.write(raw_text)

    # Spell Correction
    blob = TextBlob(raw_text)
    corrected_text = str(blob.correct())

    st.subheader("Corrected Text:")
    st.success(corrected_text)
