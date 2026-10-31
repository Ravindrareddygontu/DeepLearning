import pickle
import numpy as np
from PIL import Image
from prepare_data import ConvolutionalLayer


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

horizontal_filter = [1, 1]

c_layer = ConvolutionalLayer(horizontal_filter)
weights = pickle.load(open("weights.pkl", "rb"))

def change_image(image_path):
    image = Image.open(image_path).convert("L").resize((28, 28))
    image = np.array(image)
    for row in image:
        for i in range(len(row)):
            if row[i] > 200:
                row[i] = 0
            elif row[i] < 30:
                row[i] = 255
    Image.fromarray(image).save(image_path)

test_loss = 0
for i in range(10):
    image_path = f"test_images/{i}.jpeg"
    convolved_pixels = c_layer.convolve_image(image_path)
    output_vector = [1 if k == i else 0 for k in range(10)]
    loss, predicted_vector = predict_image(convolved_pixels, *weights, output_vector)
    test_loss += loss
    for j in range(10):
        print(i, j, predicted_vector[j])
    print("---------------------------")

print(test_loss / 10)
