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


def predict_image(pixels, weights, bias):
    affine_transform1 = np.dot(pixels, weights) + bias
    relu1 = max(0.1 * affine_transform1, affine_transform1)
    sigmoid1 = 1 / (1 + math.exp(-relu1))
    predicted_vector = [sigmoid1, 1 - sigmoid1]
    return predicted_vector


class ConvolutionalLayer:
    def __init__(self, filter):
        self.filter = filter
        self.filter_length = len(filter)

    def convolve_image(self, image_path):
        image = Image.open(image_path).convert("L")
        image = np.asarray(image)
        convolved_pixels = self.convolve_row(image)
        convolved_pixels = self.convolve_column(convolved_pixels)
        convolved_pixels = self.convolve_row(convolved_pixels)
        convolved_pixels = self.convolve_column(convolved_pixels)
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


def calculate_loss(predicted_vector, output_vector):
    predicted_vector = [max(0.0001, min(0.9999, x)) for x in predicted_vector]
    return -(output_vector[0] * math.log(predicted_vector[0]) + output_vector[1] * math.log(predicted_vector[1]))


def training(training_examples, weights1, bias):
    loss = 0
    gradients = 0
    new_weights = weights1.copy()
    for pixels, output_vector in training_examples:
        affine_transform1 = np.dot(pixels, new_weights) + bias
        relu1 = max(0.1 * affine_transform1, affine_transform1)
        sigmoid1 = 1 / (1 + math.exp(-relu1))
        predicted_vector = [sigmoid1, 1 - sigmoid1]
        loss += calculate_loss(predicted_vector, output_vector)
        gradient_loss_weights = (sigmoid1 - output_vector[0]) * (1 if affine_transform1 > 0 else 0.1)
        gradients += gradient_loss_weights
        for i, weight in enumerate(new_weights):
            gradient_loss_weight = gradient_loss_weights * pixels[i]
            new_weight = weight - 0.1 * gradient_loss_weight
            new_weights[i] = float(new_weight)
    return loss, new_weights, gradients


def prepare_train_and_test_data():
    c_layer = ConvolutionalLayer(horizontal_filter)
    training_examples = []
    test_examples = []
    for i in range(2):
        output_vector = [1 if j == i else 0 for j in range(2)]
        for file in os.listdir(f"{train_folder_path}/{i}"):
            image_path = f"{train_folder_path}/{i}/{file}"
            convolved_pixels = c_layer.convolve_image(image_path)
            training_examples.append((convolved_pixels, output_vector))
        for file in os.listdir(f"{test_folder_path}/{i}"):
            image_path = f"{test_folder_path}/{i}/{file}"
            convolved_pixels = c_layer.convolve_image(image_path)
            test_examples.append((convolved_pixels, output_vector))

    return training_examples, test_examples



# weights1 = [np.random.randn() for i in range(49)]
weights1 = [-1.6871922251633316, 0.5370027102019292, -0.9489878848670414, -1.2264839966190033, 0.6424645052648837,
            -0.3331356439119806, 0.07909875228836807, 0.04469020680054191, -1.2477961225705643, -0.8508675051533346,
            -0.8194878766280129, -0.6665420171263134, -1.2104180018859048, 0.11187474030935529, -0.5759406272570315,
            0.12584937977211538, -0.19836699740096186, 0.9461126505381419, -0.2067231156344286, 0.1667879786641714,
            0.534227403897063, 1.256324954730049, 0.48766530129462377, -1.920920809888968, 0.7730875538234291,
            0.619549288742461, 0.999958880600764, -0.7980885217258109, -1.2665244807321088, -0.4058131106736042,
            0.3548660030971968, 0.6074280138044895, 0.37544985437702744, -0.08848976399050203, -0.1224086384667234,
            -1.9068386974783016, -0.7573125191777209, 0.27028068209850103, 1.985989586579367, 1.2853067959704905,
            -1.3864542315120458, -1.6365544740799172, -0.9810587912917649, -0.13402069831847718, -0.4670510210545622,
            0.8983730156849856, -0.9599646249536022, -0.866600187367135, -1.301275790692425]


def train_and_save_weights(training_examples, test_examples, weights1):
    bias = 0
    total_error, new_weights, gradients = training(training_examples, weights1, bias)
    previous_error = total_error / len(training_examples)
    decrease_count = 0
    start_time = time.time()
    best_test_loss = 10
    best_weights = []
    best_epoch = 0

    for i in range(10000):
        total_error, new_weights, gradients = training(training_examples, new_weights, bias)
        current_error = total_error / len(training_examples)
        if decrease_count > 3:
            break
        if current_error > previous_error:
            decrease_count += 1
        previous_error = current_error
        test_loss = 0
        for convolved_pixels, output_vector in test_examples:
            predicted_vector = predict_image(convolved_pixels, new_weights, bias)
            test_loss += calculate_loss(predicted_vector, output_vector)
        test_loss = test_loss / len(test_examples)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_weights = new_weights.copy()
            best_epoch = i
        print(f"for epoch {i}, train_error: {current_error}, test_loss: {test_loss}")

    print(f"Best test loss: {best_test_loss} at epoch {best_epoch}")
    pickle.dump(new_weights, open(f"weights.pkl", "wb"))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


# training_examples, test_examples = prepare_train_and_test_data()
# train_and_save_weights(training_examples, test_examples, weights1)

