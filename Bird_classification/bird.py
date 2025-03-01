import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

training_data = keras.utils.image_dataset_from_directory(
    "Bird_Speciees_Dataset",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    validation_split=None,
    subset=None,
)

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(256,256,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(6))
model.summary()


model.compile(optimizer = 'adam',loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),metrics = ['accuracy'])
history = model.fit(training_data,epochs =10 )
