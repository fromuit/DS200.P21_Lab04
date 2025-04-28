import torchvision
from torchvision import datasets
import os
import numpy as np

DATA_DIR = "./data/mnist_raw"

def download_mnist(root_dir=DATA_DIR):
    """Downloads the MNIST dataset if not already present."""
    print(f"Downloading MNIST to {root_dir}...")
    # Use train=True just to get the main dataset file for simulation
    datasets.MNIST(root=root_dir, train=True, download=True)
    print("MNIST download complete.")

def load_mnist_data_for_streaming(root_dir=DATA_DIR):
    """
    Loads the MNIST training dataset into a list of (image_array, label) tuples.
    This is suitable for saving as stream chunks.
    """
    # We need to load the data without applying transforms initially,
    # as transforms will be applied later within the Spark process.
    mnist_dataset = datasets.MNIST(root=root_dir, train=True, download=False, transform=None)

    data_list = []
    print("Preparing MNIST data for streaming simulation...")
    for i in range(len(mnist_dataset)):
        # Get image as PIL Image and label
        image_pil, label = mnist_dataset[i]
        # Convert PIL Image to numpy array
        image_np = np.array(image_pil) # Shape (H, W), dtype uint8
        data_list.append((image_np, label))

    print(f"Prepared {len(data_list)} samples.")
    return data_list

if __name__ == '__main__':
    # Example usage
    download_mnist()
    data_for_stream = load_mnist_data_for_streaming()
    print(f"First sample image shape: {data_for_stream[0][0].shape}, dtype: {data_for_stream[0][0].dtype}")
    print(f"First sample label: {data_for_stream[0][1]}")