import json
import math
import os
import random
import pickle
import time

import numpy as np
from PIL import Image

horizontal_filter = [1, 1]
train_folder_path = "mnist_images/train"
test_folder_path = "mnist_images/test"


class ConvolutionalLayer:
    def __init__(self, filter):
        self.filter = filter
        self.filter_length = len(filter)

    def convolve_image(self, image_path):
        image = Image.open(image_path).convert("L")
        convolved_pixels = np.asarray(image)
        # convolved_pixels = self.convolve_row(image)
        # convolved_pixels = self.convolve_column(convolved_pixels)
        # Image.fromarray(convolved_pixels).save("conv1.jpg")
        # convolved_pixels = self.convolve_row(convolved_pixels)
        # convolved_pixels = self.convolve_column(convolved_pixels)
        # Image.fromarray(convolved_pixels).save("conv2.jpg")
        convolved_pixels = convolved_pixels.flatten()
        mean = np.mean(convolved_pixels)
        std = np.std(convolved_pixels)
        convolved_pixels = (convolved_pixels - mean) / std
        return convolved_pixels

    def convolve_row(self, image_object: np.ndarray):
        convolved_pixels = []
        row_length = image_object.shape[0]
        for i in range(row_length):
            row_vector = image_object[i]
            convolved_row = self.convolve(row_vector)
            convolved_pixels.append(convolved_row)
        convolved_pixels = np.array(convolved_pixels, dtype=np.uint8)
        return convolved_pixels

    def convolve(self, vector):
        vector = np.asarray(vector)
        vector_length = len(vector)
        rows = math.ceil(vector_length / self.filter_length)
        padding = rows * self.filter_length - vector_length
        padded = np.pad(vector, (0, padding))
        elems = np.reshape(padded, (rows, self.filter_length))
        result = np.dot(elems, self.filter)
        result = np.minimum(result, 255).tolist()
        return np.maximum(result, 0).tolist()

    def convolve_column(self, image_object: np.ndarray):
        convolved_pixels = []
        column_length = image_object.shape[1]
        for i in range(column_length):
            column_vector = image_object[:, i]
            convolved_column = self.convolve(column_vector)
            convolved_pixels.append(convolved_column)
        convolved_pixels = np.array(convolved_pixels, dtype=np.uint8)
        return np.transpose(convolved_pixels)


def prepare_train_and_test_data():
    c_layer = ConvolutionalLayer(horizontal_filter)
    training_examples = []
    test_examples = []
    for i in range(10):
        output_vector = [1 if j == i else 0 for j in range(10)]
        for file in os.listdir(f"{train_folder_path}/{i}"):
            image_path = f"{train_folder_path}/{i}/{file}"
            convolved_pixels = c_layer.convolve_image(image_path)
            training_examples.append((convolved_pixels, output_vector))
        for file in os.listdir(f"{test_folder_path}/{i}"):
            image_path = f"{test_folder_path}/{i}/{file}"
            convolved_pixels = c_layer.convolve_image(image_path)
            test_examples.append((convolved_pixels, output_vector))
    return training_examples, test_examples

training_examples, test_examples = prepare_train_and_test_data()
data = {
    "training": training_examples,
    "test": test_examples
}
np.save("data.npy", data)




