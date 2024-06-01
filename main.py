import os
import cv2
from random import shuffle
import numpy

DATA_DIRECTORY = "dataset/Train"
FOODS = ["Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"]

IMAGE_SIZE = 256
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
