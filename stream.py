import streamlit as st
from tensorflow import keras
import cv2
import numpy as np


Classnames= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
st.title('American Sign Language Recognition')
model = keras.models.load_model("asl.keras")
uploaded_file = st.file_uploader('upload the image')
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded ASL Image", use_column_width=True)

    # image1 = cv2.imread(image)
    img = cv2.resize(image, (128, 128))
    img_array = np.asarray(img)
    img_array = img_array.astype('float32')/ 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_class_name = Classnames[predicted_class]
    st.success(f"Predicted ASL Sign: {predicted_class_name}")


