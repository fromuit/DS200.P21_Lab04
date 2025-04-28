# stream.py
import os
import numpy as np
import glob
from PIL import Image
import torch
from pyspark import SparkContext, SparkConf
import torchvision.transforms as transforms # Cần import transform ở đây

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


STREAM_INPUT_DIR = "./data/streaming_input"
DATA_CHUNK_EXTENSION = ".npz" # Định dạng file mà receiver sẽ ghi

def load_stream_data_into_rdd(spark_context: SparkContext, stream_dir=STREAM_INPUT_DIR):
    """
    Loads data from the simulated stream directory (files written by receiver)
    into a Spark RDD.
    """
    print(f"Spark loader: Reading data from {stream_dir}...")

    # Lấy danh sách file .npz hiện có
    data_files = glob.glob(os.path.join(stream_dir, f"*{DATA_CHUNK_EXTENSION}"))
    if not data_files:
        print("No data files found in stream directory.")
        return spark_context.emptyRDD()

    # Tạo RDD từ danh sách các đường dẫn file
    file_paths_rdd = spark_context.parallelize(data_files)

    def parse_npz_file_content(file_path):
        """Helper function to load data from a single .npz file given its path."""
        try:
            # Đọc nội dung file
            with open(file_path, 'rb') as f:
                 npz_file = np.load(f, allow_pickle=True)
                 images = npz_file['images']
                 labels = npz_file['labels']
                 npz_file.close()

            # Mỗi element trong RDD sẽ là một list các (image_array, label) từ 1 chunk file
            return list(zip(images, labels))
        except Exception as e:
            print(f"Error parsing NPZ file {file_path}: {e}")
            return [] # Trả về list rỗng nếu lỗi

    # Map RDD của đường dẫn file thành RDD của (image_array, label)
    data_rdd = file_paths_rdd.flatMap(parse_npz_file_content)


    # --- Áp dụng Transform ---
    from transforms.mnist_transforms import get_mnist_transforms
    transform = get_mnist_transforms()

    def apply_transform_to_sample(sample):
        """Applies the transformation to a single (image_array, label) tuple."""
        image_np, label = sample
        try:
            # Convert numpy array (H, W, uint8) back to PIL Image
            image_pil = Image.fromarray(image_np, mode='L') # 'L' for grayscale
            # Apply torchvision transform (PIL -> Tensor [C, H, W], normalized)
            image_tensor = transform(image_pil)
            return (image_tensor, int(label)) # Return tensor and integer label
        except Exception as e:
            # print(f"Error applying transform: {e}") # Có thể bỏ comment để debug
            return None # Skip samples that fail transformation

    transformed_rdd = data_rdd.map(apply_transform_to_sample).filter(lambda x: x is not None)

    # print(f"Applied transforms. RDD now has {transformed_rdd.count()} transformed samples.")

    return transformed_rdd

if __name__ == '__main__':
    # Code test đơn giản nếu cần
    print("stream.py: Contains only data loading logic for Spark.")