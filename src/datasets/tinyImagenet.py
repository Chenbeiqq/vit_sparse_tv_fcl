import os
import torch
import zipfile
import requests
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor

class TinyImageNet:
    def __init__(self,
                 preprocess=None,
                 location=os.path.expanduser('~/data/tiny-imagenet-200'),
                 batch_size=128,
                 num_workers=12,
                 **kwargs,
                 ):

        # Define default preprocess if none is provided
        if preprocess is None:
            preprocess = Compose([
                Resize((64, 64)),  # Tiny ImageNet images are 64x64
                ToTensor(),
            ])

        # Ensure dataset is downloaded and extracted
        self.download_and_extract(location)

        # Load training dataset
        self.train_dataset = ImageFolder(
            root=os.path.join(location, 'tiny-imagenet-200/train'),
            transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # Load validation dataset
        self.test_dataset = ImageFolder(
            root=os.path.join(location, 'tiny-imagenet-200/val'),
            transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # Extract class names
        self.classnames = self.train_dataset.classes

        # Define a default class order (shuffled indices for continual learning)
        self.default_class_order = list(range(200))

    def download_and_extract(self, location):
        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = os.path.join(location, "tiny-imagenet-200.zip")

        # Create directory if it doesn't exist
        if not os.path.exists(location):
            os.makedirs(location)

        # Download the dataset if not already downloaded
        if not os.path.exists(zip_path):
            print(f"Downloading Tiny ImageNet dataset from {url}...")
            response = requests.get(url, stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")

        # Extract the dataset if not already extracted
        if not os.path.exists(os.path.join(location, 'train')):
            print("Extracting Tiny ImageNet dataset...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(location)
            print("Extraction complete.")

