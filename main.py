# main.py
import os
import time
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Chỉ cần path lên 1 cấp

from trainer import run_spark_training # Giữ nguyên import này


def main():
    print("Starting MNIST CNN training simulation on Apache Spark (reading from files filled by socket receiver).")


    # Cấu hình huấn luyện
    TRAINING_EPOCHS = 5 # Số lần Spark đọc lại dữ liệu từ thư mục file và thực hiện 1 vòng tổng hợp model
    PARTITION_EPOCHS = 1 # Local epochs trên mỗi Spark partition
    SPARK_MASTER = "local[*]" # Sử dụng tất cả core cục bộ

    # Thư mục mà receiver ghi file vào (stream input cho Spark)
    # Import từ stream.py để đảm bảo nhất quán
    from stream import STREAM_INPUT_DIR


    # --- Setup Spark và Run Training (Đọc từ file do Receiver ghi) ---
    print("\n--- Setting up Apache Spark and starting training ---")

    conf = SparkConf().setAppName("SparkSocketSimTraining").setMaster(SPARK_MASTER)
    # Cần đảm bảo các file Python (models, transforms, stream) được phân phối đến worker
    # Spark thường tự làm điều này trong chế độ local[*], nhưng trong cluster cần cấu hình thêm spark.submit.pyFiles
    # hoặc đảm bảo các file nằm trong PYTHONPATH của worker.

    sc = None
    spark = None
    try:
        sc = SparkContext(conf=conf)
        spark = SparkSession(sc)
        print("Spark session created.")

        # Run the Spark training process
        # Spark sẽ đọc từ thư mục STREAM_INPUT_DIR mà receiver_process.py đã ghi vào
        run_spark_training(
            spark_context=sc,
            spark_session=spark,
            stream_dir=STREAM_INPUT_DIR, # Vẫn đọc từ thư mục này
            epochs=TRAINING_EPOCHS,
            epochs_per_partition=PARTITION_EPOCHS,
            # Thêm model_class=ImprovedCNN nếu bạn đang dùng ImprovedCNN
            # Đảm bảo ImprovedCNN được import và cấu hình đúng trong trainer.py
            # model_class=ImprovedCNN # Ví dụ nếu dùng ImprovedCNN
        )

    except Exception as e:
        print(f"\nAn error occurred during Spark execution: {e}")
        import traceback
        traceback.print_exc() # In traceback để dễ debug
    finally:
        if spark:
            spark.stop()
            print("Spark session stopped.")
        if sc:
            sc.stop()
            print("Spark context stopped.")

    print("\nMNIST CNN training simulation finished.")

if __name__ == "__main__":
    # Khi chạy main.py trực tiếp, nó sẽ chạy Spark ngay lập tức.
    # Cần đảm bảo receiver_process.py đã chạy và ghi đủ data trước đó.
    # Tuy nhiên, mục đích chính là chạy qua run_socket_sim.py
    print("Running main.py. Ensure receiver_process.py has populated data dir.")
    main()