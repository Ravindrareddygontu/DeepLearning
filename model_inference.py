import pickle

import numpy as np
from PIL import Image

from main import ConvolutionalLayer, predict_image

horizontal_filter = [1, 1]

c_layer = ConvolutionalLayer(horizontal_filter)

image = Image.open("mnist_images/Pasted_1.png").convert("L").resize((28, 28))
image = np.array(image)
for row in image:
    for i in range(len(row)):
        if row[i] > 230:
            row[i] = 0

# image = Image.fromarray(image, 'L')
# image.save("mnist_images/Pasted_2.png")

weights = pickle.load(open("weights.pkl", "rb"))
print(weights)
convolved_pixels = c_layer.convolve_image(image_path="mnist_images/Pasted_2.png")
print(convolved_pixels)
predicted_vector = predict_image(convolved_pixels, weights, 0)
print(predicted_vector)