import os
import cv2 as cv
import streamlit as st 
import matplotlib.pyplot as plt
from keras_ocr.tools import (
    read, 
    drawAnnotations
)
from keras_ocr.pipeline import Pipeline 
from utils import ( 
    preprocess_image,
    postprocess_text
)

# caching model
# @st.cache(ttl = 115200)
def load_pipeline():
    return Pipeline()

# define base parent
BASE = "sample"

# page configuration
st.set_page_config(
    page_title = "Document OCR App",
    page_icon  = ":pancakes:",
    layout     = "wide",
)

# title dashboard
st.title("Tensorflow Document OCR")

# loading model...
with st.spinner("Load Model..."):
    model = load_pipeline()

# input widget based on sample folder
file_select = st.selectbox( 
    "Input Image Document:", 
    os.listdir("sample")
)

# read image and denoising
filename = os.path.join(BASE, file_select)
image    = cv.imread(filename)
denoised_image = preprocess_image(image)

# apply detection and recognition of document image
with st.spinner("Model Prediction..."):
    images     = [read(filename)]
    prediction = model.recognize(images)  

# show text detection and word recognition result
fig, ax = plt.subplots()
predicted_image = drawAnnotations( 
    image = denoised_image, 
    predictions = prediction[0],
    ax = ax
)
st.pyplot(fig)

# show post-processed text result
word_token = [text for text, array in prediction[0]]
text_result = postprocess_text(word_token)

st.success(f"Recognized Text : \n\n{text_result}")