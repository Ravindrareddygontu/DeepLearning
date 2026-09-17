import json

import numpy as np
from PIL import Image

from uniform_pixels import get_random_pixel, generate_uniform_pixel

with open("first_order_probabilites.json", "r") as f:
    first_order_probability_map = json.load(f)


def generate_second_pixel(pixel_seed):
    str_pixel = str(pixel_seed)
    pixel_conditionals = first_order_probability_map[str_pixel]
    sum_of_frequencies = sum(pixel_conditionals.values())
    list_of_pixels = list(pixel_conditionals.keys())
    conditional_distributions = [v / sum_of_frequencies for v in pixel_conditionals.values()]
    next_pixel = get_random_pixel(list_of_pixels, conditional_distributions)
    return next_pixel


def generate_first_order_pixels(pixel_seed):
    str_pixel = str(pixel_seed)
    pixel_conditionals = first_order_probability_map[str_pixel]
    sum_of_frequencies = sum(pixel_conditionals.values())
    list_of_pixels = list(pixel_conditionals.keys())
    conditional_distributions = [v / sum_of_frequencies for v in pixel_conditionals.values()]
    pixels = []
    for i in range(784):
        next_pixel = get_random_pixel(list_of_pixels, conditional_distributions)
        pixels.append(int(next_pixel))
    return pixels

# random_pixel = generate_uniform_pixel()
# generated_pixels = generate_first_order_pixels(random_pixel)
# image = Image.fromarray(np.array(generated_pixels, dtype=np.uint8).reshape(28, 28), 'L')
# image.save("first_order_images/image5.png")

