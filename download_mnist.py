import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import os

API_URL = "https://datasets-server.huggingface.co/rows"
OUTPUT_DIR = "mnist_images"

# Create digit folders
for digit in range(10):
    os.makedirs(f"{OUTPUT_DIR}/{digit}", exist_ok=True)


def download_image(item):
    index = item["row_idx"]
    label = item["row"]["label"]
    image_url = item["row"]["image"]["src"]

    response = requests.get(image_url)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))

    filename = f"{OUTPUT_DIR}/{label}/{index}.jpg"
    image.save(filename)

    return index


for offset in range(0, 10000, 100):

    params = {
        "dataset": "ylecun/mnist",
        "config": "mnist",
        "split": "train",
        "offset": offset,
        "length": 100
    }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()

    rows = response.json()["rows"]

    # Download 20 images simultaneously
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(download_image, rows))

    print(f"Downloaded {offset + len(rows)}/10000")

print("Done!")