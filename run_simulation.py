# run_simulation.py
import subprocess
import time
import os
import sys
import multiprocessing

import receiver_process
import sender_process
# Add parent directory to path to allow imports (cho các script con)
# Không cần ở đây nếu chạy bằng subprocess với python -m
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_process(script_name, process_name):
    """Helper function to run a Python script in a separate process."""
    print(f"--- Starting {process_name} ({script_name}) ---")
    # Sử dụng python -m để chạy script, giúp import các module khác trong cùng package dễ hơn
    process = subprocess.Popen([sys.executable, "-u", script_name], stdout=sys.stdout, stderr=sys.stderr)
    # Ghi stdout/stderr ra console hiện tại để dễ theo dõi

    # Có thể trả về process object nếu cần quản lý (kill, wait)
    return process

def run_process_mp(target_function, process_name, *args):
     """Helper function to run a function in a separate process using multiprocessing."""
     print(f"--- Starting {process_name} ---")
     process = multiprocessing.Process(target=target_function, args=args)
     process.start()
     return process


if __name__ == '__main__':
    # Thiết lập môi trường (đặt biến môi trường cho Spark nếu cần)
    # Đảm bảo biến môi trường PYSPARK_PYTHON và PYSPARK_DRIVER_PYTHON được đặt
    # trước khi chạy script này hoặc trong môi trường shell/terminal bạn dùng.
    # Ví dụ:
    # import os
    # os.environ['PYSPARK_PYTHON'] = 'C:\DuongDanDenFilePythonCuaBan\python.exe'
    # os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:\DuongDanDenFilePythonCuaBan\python.exe'
    # print("Environment variables for Spark set.")


    # Xóa thư mục output cũ nếu cần
    stream_dir = "./data/streaming_input"
    print(f"Clearing previous data in {stream_dir}...")
    # Không xóa thư mục, chỉ xóa file bên trong
    # import glob
    # for f in glob.glob(os.path.join(stream_dir, "*")):
    #     os.remove(f)
    # Việc xóa file đã được thêm vào receiver_process.py lúc khởi động

    receiver_proc = None
    sender_proc = None
    spark_proc = None

    try:
        # 1. Khởi động Receiver Process (phải chạy trước Sender)
        # receiver_proc = run_process("receiver_process.py", "Receiver")
        receiver_proc = run_process_mp(receiver_process.run_receiver, "Receiver", 1000) # Receiver ghi 1000 mẫu/file

        # Đợi Receiver khởi động và sẵn sàng lắng nghe
        print("Waiting for Receiver to start...")
        time.sleep(5) # Đợi 5 giây để Receiver mở socket

        # 2. Khởi động Sender Process
        # sender_proc = run_process("sender_process.py", "Sender")
        sender_proc = run_process_mp(sender_process.run_sender, "Sender", 20000, 500) # Sender gửi 20000 mẫu, mỗi lần gửi 500 mẫu

        # Đợi Sender hoàn thành (hoặc chạy song song với Spark)
        # Để đơn giản, chúng ta đợi sender hoàn thành trước khi chạy Spark
        # Trong hệ thống thực, Spark Streaming sẽ chạy liên tục và xử lý dữ liệu khi nó đến
        print("Waiting for Sender to finish sending data...")
        sender_proc.join() # Đợi tiến trình sender kết thúc
        print("Sender process finished.")

        # Spark sẽ đọc dữ liệu từ các file mà Receiver đã ghi ra
        # Lúc này, các file đã được tạo xong bởi Receiver (vì sender đã finish)
        # Trong mô phỏng streaming nâng cao hơn, Spark sẽ chạy song song và đọc các file mới khi chúng xuất hiện

        # Đợi thêm chút cho Receiver ghi nốt các mẫu cuối cùng ra file nếu có
        time.sleep(2)
        print("Ready to start Spark training...")

        # 3. Khởi động Main Spark Training Process
        # spark_proc = run_process("main.py", "Spark Trainer")
        # Chạy main.py trong tiến trình hiện tại hoặc dùng subprocess/multiprocessing
        # Chạy trực tiếp trong tiến trình này đơn giản hơn nếu không cần quản lý riêng
        # Nhưng nếu main.py gọi sc.stop() cuối cùng, nó sẽ đóng luôn SparkContext.
        # Dùng subprocess an toàn hơn nếu main.py không có cơ chế dừng đặc biệt.
        # Tuy nhiên, main.py của chúng ta đã có try/finally để stop Spark.
        # Vẫn nên dùng subprocess để giả lập 3 tiến trình riêng biệt hơn.
        spark_proc = run_process(os.path.join(".", "main.py"), "Spark Trainer") # Đường dẫn tương đối đến main.py
        spark_proc.wait() # Đợi Spark training kết thúc

    except KeyboardInterrupt:
        print("\nShutdown initiated by user.")
    except Exception as e:
        print(f"\nAn error occurred during orchestration: {e}")