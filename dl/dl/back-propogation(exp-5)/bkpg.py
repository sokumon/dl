import pandas as pd
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv("placementdataset.csv")

df.head()

df.shape

df.info()

df.duplicated().sum()

df = df.drop_duplicates()

df.shape

X = df[["CGPA", "Profile_Score"]]
y = df["Placed"]

y

X.head()

model = Sequential()
model.add(Dense(4, input_dim=2, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.get_weights()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(X, y, epochs=100, batch_size=1, verbose=1)

loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

import matplotlib.pyplot as plt

plt.plot(history.history["loss"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

import numpy as np

new_data = np.array([[8.1, 6.1]])
prediction = model.predict([new_data])

prediction

prediction_binary = (prediction > 0.5).astype(int)
print("Prediction:", prediction_binary)
