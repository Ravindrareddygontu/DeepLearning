import pickle
import time

import numpy as np


start = time.perf_counter()

learning_rate = 0.0001
batch_size = 64

def predict_image(pixels, hidden_layer_weights1, hidden_layer_weights2, output_layer_weights, hidden_layer_bias1, hidden_layer_bias2, output_layer_bias, output_vector):
    affine = (hidden_layer_weights1 @ pixels) + hidden_layer_bias1
    relus = np.maximum(0.1 * affine, affine)
    affine2 = hidden_layer_weights2 @ relus + hidden_layer_bias2
    relus2 = np.maximum(0.1 * affine2, affine2)
    logits = (output_layer_weights @ relus2) + output_layer_bias
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    predicted_vector = exp_logits / np.sum(exp_logits)
    loss = np.log(np.sum(exp_logits)) - np.dot(shifted_logits, output_vector)
    return loss, predicted_vector


def initialize_zeros(like):
    return np.zeros_like(like)


def initialize_weights(rows, cols):
    slope = 0.1
    scale = np.sqrt(2 / ((1 + slope ** 2) * cols))
    return np.random.randn(rows, cols) * scale


def shuffle_images(training_examples):
    no_of_examples = len(training_examples)
    indices = np.random.permutation(no_of_examples)
    shuffled_training_examples = [training_examples[i] for i in indices]
    shuffled_pixels = []
    shuffled_outputs = []
    for example in shuffled_training_examples:
        shuffled_pixels.append(example[0])
        shuffled_outputs.append(example[1])
    batch_training_examples = []
    batch_outputs = []
    for i in range(0, no_of_examples, batch_size):
        batch_training_examples.append(shuffled_pixels[i:i+batch_size])
        batch_outputs.append(shuffled_outputs[i:i+batch_size])
    return batch_training_examples, batch_outputs


def training_multiple():
    data = np.load("data.npy", allow_pickle=True).item()
    training_examples, test_examples = data["training"], data["test"]
    no_of_examples = len(training_examples)
    hidden_layer_neurons1, hidden_layer_neurons2, output_layer_neurons = 30, 30, 10
    input_size = len(training_examples[0][0])
    print("input size: ", input_size, "no of examples: ", no_of_examples)
    hidden_layer_weights1 = initialize_weights(hidden_layer_neurons1, input_size)
    hidden_layer_weights2 = initialize_weights(hidden_layer_neurons2, hidden_layer_neurons1)
    output_layer_weights = (np.random.randn(10, 30) * np.sqrt(2 / (30 + 10)))
    hidden_layer_bias1 = np.zeros(hidden_layer_neurons1)
    hidden_layer_bias2 = np.zeros(hidden_layer_neurons2)
    output_layer_bias = np.zeros(output_layer_neurons)
    best_test_loss = best_train_loss = 1000
    for epoch in range(10000):
        loss = 0
        test_loss = 0
        batch_training_examples, batch_outputs = shuffle_images(training_examples)
        for i, batch_examples in enumerate(batch_training_examples):
            outputs = batch_outputs[i]
            affine1 = batch_examples @ hidden_layer_weights1.T + hidden_layer_bias1
            relu1 = np.maximum(0.1 * affine1, affine1)
            affine2 = relu1 @ hidden_layer_weights2.T + hidden_layer_bias2
            relu2 = np.maximum(0.1 * affine2, affine2)
            logits = relu2 @ output_layer_weights.T + output_layer_bias
            shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(shifted_logits)
            sum_logits = np.sum(exp_logits, axis=1)
            predicted_vector = exp_logits / sum_logits[:, np.newaxis]
            loss += np.sum(np.log(sum_logits) - np.sum(outputs * shifted_logits, axis=1))
            loss_derivative_wrt_logits = predicted_vector - outputs
            output_layer_gradients = loss_derivative_wrt_logits.T @ relu2
            output_layer_bias_gradients = loss_derivative_wrt_logits.sum(axis=0)
            hidden_derivatives2 = loss_derivative_wrt_logits @ output_layer_weights
            hidden_derivatives2 = hidden_derivatives2 * np.where(affine2 < 0, 0.1, 1)
            hidden_layer_gradients2 = hidden_derivatives2.T @ relu1
            hidden_layer_bias_gradients2 = hidden_derivatives2.sum(axis=0)
            hidden_derivatives1 = hidden_derivatives2 @ hidden_layer_weights2
            hidden_derivatives1 = hidden_derivatives1 * np.where(affine1 < 0, 0.1, 1)
            hidden_layer_gradients1 = hidden_derivatives1.T @ batch_examples
            hidden_layer_bias_gradients1 = hidden_derivatives1.sum(axis=0)
            len_random_examples = len(batch_examples)
            hidden_layer_weights1 = update_weights_matrix(hidden_layer_weights1, hidden_layer_gradients1, len_random_examples)
            hidden_layer_weights2 = update_weights_matrix(hidden_layer_weights2, hidden_layer_gradients2, len_random_examples)
            output_layer_weights = update_weights_matrix(output_layer_weights, output_layer_gradients, len_random_examples)
            hidden_layer_bias1 = update_bias_vectors(hidden_layer_bias1, hidden_layer_bias_gradients1, len_random_examples)
            hidden_layer_bias2 = update_bias_vectors(hidden_layer_bias2, hidden_layer_bias_gradients2, len_random_examples)
            output_layer_bias = update_bias_vectors(output_layer_bias, output_layer_bias_gradients, len_random_examples)
        for convolved_pixels, output_vector in test_examples:
            t_loss, _ = predict_image(convolved_pixels, hidden_layer_weights1, hidden_layer_weights2, output_layer_weights, hidden_layer_bias1, hidden_layer_bias2, output_layer_bias, output_vector)
            test_loss += t_loss
        test_loss = test_loss / len(test_examples)
        train_loss = loss / len(training_examples)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_loss = train_loss
            best_weights = (hidden_layer_weights1, hidden_layer_weights2, output_layer_weights, hidden_layer_bias1, hidden_layer_bias2, output_layer_bias)
            pickle.dump(best_weights, open(f"weights.pkl", "wb"))
        print(f"Epoch {epoch}: Training loss: {train_loss}, Test loss: {test_loss}, Best test loss: {best_test_loss}, Best train loss: {best_train_loss}, Time elapsed: {time.perf_counter() - start}")
    print(f"Time taken: {time.perf_counter() - start} seconds")


def update_weights_matrix(weight_matrix, gradient_matrix, no_of_examples):
    gradient_matrix = gradient_matrix / no_of_examples
    # gradient_matrix = learning_rate * gradient_matrix
    # gradient_matrix = learning_rate * (gradient_matrix + 0.01 * np.sign(weight_matrix))
    gradient_matrix = learning_rate * (gradient_matrix + 0.001 * 2 * weight_matrix)
    weight_matrix = weight_matrix - gradient_matrix
    return weight_matrix


def update_bias_vectors(weight_matrix, gradient_matrix, no_of_examples):
    gradient_matrix = gradient_matrix / no_of_examples
    gradient_matrix = learning_rate * gradient_matrix
    weight_matrix = weight_matrix - gradient_matrix
    return weight_matrix


def calculate_derivatives(first, second):
    derivatives = []
    for i in range(len(first)):
        derivative = second[i] - first[i]
        derivatives.append(derivative)
    return derivatives


training_multiple()



