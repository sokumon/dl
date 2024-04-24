# Dataset - https://www.kaggle.com/datasets/salader/dogs-vs-cats

# To speed up training go to Runtime select Change Runtime type to GPU or TPU

# go to your Kaggle profile and create a new token download from kaggle and upload to current working directory

# !mkdir -p ~/.kaggle
# !cp /content/kaggle.json ~/.kaggle/

# !kaggle datasets download -d salader/dogs-vs-cats

import zipfile

zip_ref = zipfile.ZipFile("/content/dogs-vs-cats.zip", "r")
zip_ref.extractall("/content")
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
    Dropout,
)  # BN and DO are used to increase accuracy

# generators divides the data into batches to increase the speed and use RAM effectively
# gererators are very useful to process large amount of data
# detailed documentation of generators https://keras.io/api/data_loading/image/
train_ds = keras.utils.image_dataset_from_directory(
    directory="/content/train",  # path of Train folder
    labels="inferred",
    label_mode="int",  # assign 0 for cat and 1 for dog
    batch_size=32,
    image_size=(256, 256),  # reshape images to 256*256*3
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory="/content/test",  # path of Test folder
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256, 256),
)


# Normalize to values from 0-255 to 0-1
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label


train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# create CNN model
# CNN Architecture - 3 Convolutional Layers - in first layer 32 filters - in second layer 64 filters - and in third layer 128 filters
# Pooling layer for Dimensionality Reduction & Translation Invariance

model = Sequential()

model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        padding="valid",
        activation="relu",
        input_shape=(256, 256, 3),
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

model.add(Conv2D(64, kernel_size=(3, 3), padding="valid", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

model.add(Conv2D(128, kernel_size=(3, 3), padding="valid", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)

import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], color="red", label="train")
plt.plot(history.history["val_accuracy"], color="blue", label="validation")
plt.legend()
plt.show()
# try to decrease the gap to reduce overfitting

plt.plot(history.history["loss"], color="red", label="train")
plt.plot(history.history["val_loss"], color="blue", label="validation")
plt.legend()
plt.show()
# try to decrease the gap to reduce overfitting

import cv2
import matplotlib.pyplot as plt

test_img1 = cv2.imread("/content/Dog.jpg")

plt.imshow(test_img1)

test_img1.shape  # actual shape of the image

test_img1 = cv2.resize(test_img1, (256, 256))

test_input1 = test_img1.reshape(
    (1, 256, 256, 3)
)  # in this batch there is only one image

model.predict(test_input1)  # classn 0 for Cat and 1 for Dog

test_img2 = cv2.imread("/content/Cat.jpg")

plt.imshow(test_img2)

test_img2.shape

test_img2 = cv2.resize(test_img2, (256, 256))

test_input2 = test_img2.reshape((1, 256, 256, 3))

model.predict(test_input2)
