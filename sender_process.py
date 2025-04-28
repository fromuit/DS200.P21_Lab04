# sender_process.py
import socket
import pickle
import time
import numpy as np
import os

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader import load_mnist_data_for_streaming # Tái sử dụng dataloader

HOST = 'localhost'  
PORT = 9999         # Cổng mà receiver lắng nghe

def send_data_chunk(data_list):
    """Sends a list of (image_np, label) samples over the socket."""
    client_socket = None
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print(f"Sender: Connected to {HOST}:{PORT}")

        # Gửi từng mẫu một (hoặc có thể nhóm lại thành batch lớn hơn trước khi gửi)
        for i, (image_np, label) in enumerate(data_list):
            # Serialize dữ liệu (numpy array và label)
            data_to_send = (image_np, label)
            serialized_data = pickle.dumps(data_to_send)

            # Lấy kích thước dữ liệu đã serialize
            data_size = len(serialized_data)

            # Gửi kích thước (sử dụng 4 bytes - integer)
            # Cần endianness nhất quán (network byte order là big-endian)
            size_bytes = data_size.to_bytes(4, byteorder='big')
            client_socket.sendall(size_bytes)

            # Gửi dữ liệu thật
            client_socket.sendall(serialized_data)

            # print(f"Sender: Sent sample {i+1}/{len(data_list)} (size: {data_size} bytes)")

            # Mô phỏng stream thời gian thực bằng cách chờ giữa các mẫu hoặc batch
            # time.sleep(0.01) # Delay nhỏ giữa các mẫu

        print(f"Sender: Finished sending {len(data_list)} samples.")

    except ConnectionRefusedError:
        print(f"Sender Error: Connection refused. Is the receiver running on {HOST}:{PORT}?")
    except Exception as e:
        print(f"Sender Error: {e}")
    finally:
        if client_socket:
            client_socket.close()
            print("Sender: Socket closed.")

def run_sender(num_samples_to_send=10000, samples_per_send_batch=100):
    """Loads data and sends it in batches."""
    print("Sender: Loading MNIST data...")
    all_data = load_mnist_data_for_streaming() # List of (np_image, label)

    if num_samples_to_send > len(all_data):
        print(f"Sender Warning: Requested {num_samples_to_send} samples but only {len(all_data)} available. Sending all available data.")
        num_samples_to_send = len(all_data)

    print(f"Sender: Preparing to send {num_samples_to_send} samples...")

    sent_count = 0
    while sent_count < num_samples_to_send:
        batch_start = sent_count
        batch_end = min(sent_count + samples_per_send_batch, num_samples_to_send)
        data_batch = all_data[batch_start:batch_end]

        if not data_batch:
            break # Hết dữ liệu

        print(f"Sender: Sending batch of {len(data_batch)} samples (total sent: {sent_count}/{num_samples_to_send})...")
        send_data_chunk(data_batch)

        sent_count += len(data_batch)

        # Delay giữa các batch gửi
        time.sleep(0.5) # Delay 0.5 giây giữa các batch lớn hơn

    print("Sender: Data sending simulation finished.")

if __name__ == '__main__':
    # Khi chạy riêng lẻ để test:
    print("Running Sender Process standalone. Ensure Receiver is running.")
    run_sender(num_samples_to_send=1000) # Gửi 1000 mẫu để test