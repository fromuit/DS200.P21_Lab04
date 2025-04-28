import torchvision.transforms as transforms

# Define standard MNIST transformations
def get_mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST mean and std
    ])

if __name__ == '__main__':
    # Example usage (requires a dummy PIL Image)
    from PIL import Image
    import numpy as np

    # Create a dummy 28x28 grayscale image (e.g., a white square)
    dummy_image_np = np.ones((28, 28), dtype=np.uint8) * 255
    dummy_image_pil = Image.fromarray(dummy_image_np, mode='L')

    transform = get_mnist_transforms()
    transformed_tensor = transform(dummy_image_pil)

    print(f"Original image mode: {dummy_image_pil.mode}")
    print(f"Transformed tensor shape: {transformed_tensor.shape}")
    print(f"Transformed tensor dtype: {transformed_tensor.dtype}")
    print(f"Transformed tensor min: {transformed_tensor.min()}")
    print(f"Transformed tensor max: {transformed_tensor.max()}")

    # Note: ToTensor converts PIL Image to [C, H, W] float tensor [0, 1]
    # Normalize then adjusts the range based on mean/std