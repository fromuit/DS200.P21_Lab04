# receiver_process.py
import socket
import pickle
import numpy as np
import os
import time
import glob

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import STREAM_INPUT_DIR từ stream.py
from stream import STREAM_INPUT_DIR, DATA_CHUNK_EXTENSION

HOST = 'localhost'  
PORT = 9999         # Cổng để lắng nghe

# Ensure directory exists for saving received data
os.makedirs(STREAM_INPUT_DIR, exist_ok=True)

def receive_all(sock, n_bytes):
    """Helper function to ensure all n_bytes are received."""
    data = bytearray()
    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))
        if not packet:
            return None # Connection closed
        data.extend(packet)
    return data

def run_receiver(samples_per_file_chunk=500):
    """Runs the receiver server to listen for data and save it to files."""
    server_socket = None
    file_chunk_counter = 0
    current_file_samples = []

    # Xóa các file cũ trong thư mục nhận trước khi bắt đầu
    print(f"Receiver: Clearing previous data in {STREAM_INPUT_DIR}...")
    for f in glob.glob(os.path.join(STREAM_INPUT_DIR, f"*{DATA_CHUNK_EXTENSION}")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Receiver Warning: Could not remove file {f}: {e}")


    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Cho phép tái sử dụng địa chỉ socket ngay sau khi đóng
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_socket.bind((HOST, PORT))
        server_socket.listen(1) # Chỉ chấp nhận 1 kết nối tại 1 thời điểm
        print(f"Receiver: Listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"Receiver: Accepted connection from {addr}")

            try:
                while True:
                    # 1. Nhận kích thước dữ liệu (4 bytes integer)
                    size_bytes = receive_all(conn, 4)
                    if size_bytes is None: # Kết nối đóng
                        print(f"Receiver: Connection closed by {addr}")
                        break

                    data_size = int.from_bytes(size_bytes, byteorder='big')
                    # print(f"Receiver: Receiving data of size {data_size} bytes...")

                    # 2. Nhận dữ liệu thật
                    serialized_data = receive_all(conn, data_size)
                    if serialized_data is None: # Kết nối đóng
                         print(f"Receiver: Connection closed by {addr}")
                         break

                    # 3. Deserialize dữ liệu
                    data = pickle.loads(serialized_data)
                    # data is now (image_np, label) tuple

                    # 4. Lưu dữ liệu nhận được vào buffer file chunk
                    current_file_samples.append(data)

                    # print(f"Receiver: Received one sample. Buffer size: {len(current_file_samples)}")

                    # 5. Khi đủ mẫu cho một file chunk, ghi ra đĩa
                    if len(current_file_samples) >= samples_per_file_chunk:
                        chunk_filename = os.path.join(STREAM_INPUT_DIR, f"chunk_{file_chunk_counter:04d}_{int(time.time())}{DATA_CHUNK_EXTENSION}")
                        images_np = [item[0] for item in current_file_samples]
                        labels_np = [item[1] for item in current_file_samples]

                        np.savez_compressed(chunk_filename, images=images_np, labels=labels_np)
                        print(f"Receiver: Saved chunk file {file_chunk_counter+1} ({len(current_file_samples)} samples) to {chunk_filename}")

                        current_file_samples = [] # Reset buffer
                        file_chunk_counter += 1

            except Exception as e:
                print(f"Receiver Error during communication with {addr}: {e}")
            finally:
                conn.close()
                print(f"Receiver: Connection with {addr} closed.")

    except KeyboardInterrupt:
        print("Receiver: Shutting down...")
    except Exception as e:
        print(f"Receiver Error: {e}")
    finally:
        # Ghi nốt các mẫu còn lại vào file khi receiver tắt
        if current_file_samples:
            chunk_filename = os.path.join(STREAM_INPUT_DIR, f"chunk_{file_chunk_counter:04d}_final_{int(time.time())}{DATA_CHUNK_EXTENSION}")
            images_np = [item[0] for item in current_file_samples]
            labels_np = [item[1] for item in current_file_samples]
            np.savez_compressed(chunk_filename, images=images_np, labels=labels_np)
            print(f"Receiver: Saved final chunk file ({len(current_file_samples)} samples) to {chunk_filename}")

        if server_socket:
            server_socket.close()
            print("Receiver: Server socket closed.")


if __name__ == '__main__':
    # Khi chạy riêng lẻ để test:
    print("Running Receiver Process standalone.")
    run_receiver(samples_per_file_chunk=500) # Lưu 500 mẫu vào 1 file .npz