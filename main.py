import os
import cv2
from random import shuffle
import numpy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle

DATA_DIRECTORY = "dataset/Train"
FOODS = ["Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"]

IMAGE_SIZE = 50
training_data = []

for food_index, food in enumerate(FOODS):
    path = os.path.join(DATA_DIRECTORY, food)

    for image in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, image))
        image_array = cv2.resize(image_array, (IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([image_array, food_index])

shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = numpy.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
y = numpy.array(y)

X = X / 255
y = to_categorical(y, num_classes=10)

'''model = Sequential([
    Conv2D(64, (3,3), input_shape=X.shape[1:], activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(64, activation="relu"),
    
    Dense(10, activation="softmax"),
    ])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

with open("model.pickle", "wb") as file:
    pickle.dump(model, file)'''

saved_model = open("model.pickle", "rb")
model = pickle.load(saved_model)
