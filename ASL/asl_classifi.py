from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import matplotlib.pyplot as plt

data = 'asl_dataset'

train=keras.utils.image_dataset_from_directory(data,
                                         labels='inferred',
                                         label_mode='categorical',
                                         color_mode='rgb',
                                         batch_size=32,
                                         image_size=(128,128),
                                         shuffle = True,
                                         seed = 1,
                                         validation_split=0.2,
                                         subset='training')

valid=keras.utils.image_dataset_from_directory(data,
                                         labels='inferred',
                                         label_mode='categorical',
                                         color_mode='rgb',
                                         batch_size=32,
                                         image_size=(128,128),
                                         shuffle = True,
                                         seed = 1,
                                         validation_split=0.2,
                                         subset='validation')

num_classes = len(train.class_names)
# Get the class names from the training dataset
class_names = train.class_names
print("Class names:", class_names)


callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model = Sequential(
    [
        Conv2D(64,kernel_size=3,activation='relu',input_shape=(128,128,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(32,kernel_size=3,activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(32,activation='relu',kernel_regularizer='l2'),
        Dropout(0.2),
        # Dense(18,activation='relu'),
        Dense(10,activation='relu',kernel_regularizer='l2'),
        Dense(num_classes,activation='softmax')


    ]
)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train, epochs=100, validation_data=valid, callbacks=[callback])
model.save('asl.keras')



# Extract loss values from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot loss curves
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
plt.plot(val_loss, label='Validation Loss', color='red', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()


