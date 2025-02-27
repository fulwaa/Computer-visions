from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
Classnames= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
model = keras.models.load_model('asl.keras')
image = cv2.imread('images(1).jpeg')
img = cv2.resize(image, (128, 128))
img_array = np.asarray(img)
img_array = img_array.astype('float32')
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print(predicted_class)
predicted_class_name = Classnames[predicted_class]
print(f"Predicted ASL Sign: {predicted_class_name}")



