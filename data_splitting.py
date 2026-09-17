import os

folder_path = "mnist_images"
train_folder = "mnist_images/train"
test_folder = "mnist_images/test"

train_ratio = 0.8
test_ratio = 0.2

def split_data():
    for folder in os.listdir(folder_path):
        if folder == "train" or folder == "test":
            continue
        folder = os.path.join(folder_path, folder)
        no_of_images = len(os.listdir(folder))
        train_count = int(no_of_images * train_ratio)
        test_count = int(no_of_images * test_ratio)
        print(f"Folder: {folder}, Train: {train_count}, Test: {test_count}")
        for i, image in enumerate(os.listdir(folder)):
            if i < train_count:
                os.makedirs(os.path.join(train_folder, folder.split("/")[-1]), exist_ok=True)
                os.rename(os.path.join(folder, image), os.path.join(train_folder, folder.split("/")[-1], image))
            else:
                os.makedirs(os.path.join(test_folder, folder.split("/")[-1]), exist_ok=True)
                os.rename(os.path.join(folder, image), os.path.join(test_folder, folder.split("/")[-1], image))


