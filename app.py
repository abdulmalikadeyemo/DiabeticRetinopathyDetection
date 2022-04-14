from fastai.vision import *
from fastai.imports import *
from fastai.learner import *
# from fastai.vision.all import *

from fastai.vision.all import *
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
from PIL import Image
import requests
from io import BytesIO
import urllib.request


def get_x(r): return image_path/r['train_image_name']
def get_y(r): return r['class']

st.title("Diabetic Retinopathy Detection System")

path = Path()

@st.cache(allow_output_mutation=True)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1hHcW9HJ4uEh4MYrZgqKi6G1HfzYdZaGB"
urllib.request.urlretrieve(MODEL_URL, "export.pkl")
model = load_learner(Path("."), "export.pkl", cpu=True)

# learner = load_learner(Path("."), "export.pkl")

def predict(img, display_img):
    
    # Display the test image
    st.image(display_img, use_column_width=True)

    # Temporarily displays a message while executing 
    with st.spinner('Wait for it...'):
        
        time.sleep(3)      
  
  # Load model and make prediction
#     model = load_learner(path/'export.pkl', cpu=True)
    
    pred_class = model.predict(img)[0] # get the predicted class
    pred_prob = round(torch.max(model.predict(img)[2]).item()*100) # get the max probability

    # Display the prediction
    if str(pred_class) == 1:
        st.success("Presence of Diabetic Retinopathy with" + str(pred_prob) + '%. confidence')
    else:
        st.success("Absence of Diabetic Retinopathy with" + str(pred_prob) + '%. confidence')
        
        

# Image source selection
option = st.radio('', ['Choose a test image'])

if option == 'Choose a test image':

    # Test image selection
    test_images = os.listdir(path/'data/sample/')
    test_image = st.selectbox(
        'Please select a test image:', test_images)

    # Read the image
    file_path = path/'data/sample'/test_image
    img = PILImage.create(file_path)
    # Get the image to display
    display_img = mpimg.imread(file_path)

    # Predict and display the image
    predict(img, display_img)
