import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

"""MNIST is builtin dataset of keras"""

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train.shape

X_train

X_train[0]

X_test.shape

y_train

import matplotlib.pyplot as plt

plt.imshow(X_train[0])

plt.imshow(X_train[1])

plt.imshow(X_train[2])

X_train = X_train / 255
X_test = X_test / 255

X_train[0]

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
)

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

model.predict(X_test)

y_prob = model.predict(X_test)

y_prob

y_prob.argmax(axis=1)

y_pred = y_prob.argmax(axis=1)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

X_test[0]

plt.imshow(X_test[0])

model.predict(X_test[0].reshape(1, 28, 28))

model.predict(X_test[0].reshape(1, 28, 28)).argmax(axis=1)

plt.imshow(X_test[1])

model.predict(X_test[1].reshape(1, 28, 28))

model.predict(X_test[1].reshape(1, 28, 28)).argmax(axis=1)

plt.imshow(X_test[2])

model.predict(X_test[2].reshape(1, 28, 28))

model.predict(X_test[2].reshape(1, 28, 28)).argmax(axis=1)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
