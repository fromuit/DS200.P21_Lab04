# Mô Phỏng Huấn Luyện CNN trên MNIST bằng Apache Spark (Qua Socket)

Dự án này mô phỏng quá trình huấn luyện một Mạng Nơ-ron Tích Chập (CNN) trên bộ dữ liệu chữ số viết tay MNIST sử dụng Apache Spark. Điểm đặc biệt của mô phỏng này là dữ liệu không được Spark đọc trực tiếp từ nguồn gốc, mà được truyền từ một **tiến trình gửi dữ liệu (sender)** đến một **tiến trình nhận dữ liệu (receiver)** thông qua **network socket (TCP)**. Tiến trình nhận sau đó sẽ ghi dữ liệu nhận được vào một thư mục, và Spark sẽ đọc dữ liệu từ thư mục này để thực hiện huấn luyện.

## Cấu Trúc Dự Án

Dự án được tổ chức thành các file Python với vai trò cụ thể:

- **main.py**: Chứa logic khởi tạo Spark Session và bắt đầu quy trình huấn luyện chính bằng cách gọi `trainer.py`. File này đóng vai trò là điểm vào cho tiến trình huấn luyện Spark, được khởi động bởi `run_socket_sim.py`.
- **trainer.py**: Chứa hàm `run_spark_training` thực hiện vòng lặp huấn luyện trên Spark. Nó sử dụng dữ liệu từ RDD được cung cấp (đọc từ thư mục do receiver ghi) và áp dụng logic huấn luyện CNN phân tán đơn giản (huấn luyện trên partition và tổng hợp model state_dict).
- **stream.py**: Chứa hàm `load_stream_data_into_rdd` làm nhiệm vụ đọc các file dữ liệu (đã được tiến trình receiver ghi) từ thư mục cấu hình và chuyển chúng thành Spark RDD sẵn sàng cho quá trình huấn luyện/đánh giá. File này cũng định nghĩa đường dẫn thư mục input stream.
- **receiver_process.py**: **Tiến trình riêng biệt thứ nhất.** Mở một socket server, lắng nghe kết nối. Khi nhận được kết nối từ sender, nó nhận các gói dữ liệu qua socket, deserialize chúng và **ghi các mẫu dữ liệu nhận được thành các file** trong thư mục cấu hình (thư mục mà `stream.py` đọc).
- **sender_process.py**: **Tiến trình riêng biệt thứ hai.** Tải bộ dữ liệu MNIST gốc, mở một socket client, kết nối tới receiver. Nó đọc các mẫu dữ liệu MNIST, serialize chúng và **gửi qua socket** tới receiver theo từng đợt, có mô phỏng độ trễ.
- **run_simulation.py**: **Điểm khởi chạy chính cho toàn bộ mô phỏng socket.** Script này chịu trách nhiệm khởi động các tiến trình `receiver_process.py`, `sender_process.py` (theo đúng thứ tự và có thể thêm độ trễ), và sau đó khởi động tiến trình `main.py` (chạy Spark). Nó điều phối toàn bộ luồng mô phỏng từ gửi socket đến huấn luyện Spark.
- **models/cnn.py**: Định nghĩa kiến trúc của mô hình Mạng Nơ-ron Tích chập (CNN) (`SimpleCNN`).
- **transforms/mnist_transforms.py**: Định nghĩa các phép biến đổi dữ liệu cho ảnh MNIST (ToTensor, Normalize).
- **dataloader.py**: Chứa logic để tải bộ dữ liệu MNIST gốc (sử dụng `torchvision`), được sử dụng bởi `sender_process.py`.


## Yêu Cầu


- **Python**: Phiên bản từ 3.8 đến 3.11.
- **Apache Spark**: Đã cài đặt và cấu hình. Bạn cần có PySpark.
- **Môi trường chạy Java (JRE)**: Spark yêu cầu Java để hoạt động (chưa thử JDK).
- Các thư viện Python cần thiết (cài đặt bằng lệnh `pip install -r requirements.txt`)
- Code này được biên và thực thi trên môi trường Windows.

## Cách Chạy
Scripts receiver_process.py, sender_process.py, và main.py cần được chạy dưới dạng các tiến trình riêng biệt để mô phỏng đúng luồng. Script run_socket_sim.py được cung cấp để tự động hóa việc khởi động các tiến trình này theo đúng thứ tự.

1. **Thiết lập biến môi trường cho Spark:** Đảm bảo các biến môi trường **PYSPARK_PYTHON** và **PYSPARK_DRIVER_PYTHON** được đặt trỏ đến file thực thi Python của bạn (ví dụ: C:\Users\$username$\AppData\Local\Programs\Python\Python3x\python.exe). Điều này rất quan trọng để Spark có thể gọi đúng trình thông dịch Python (Đã gặp lỗi và fix được bằng cách này).

2. **Khởi động toàn bộ quá trình mô phỏng:** Mở Command Prompt hoặc PowerShell, điều hướng đến thư mục gốc của dự án và chạy script điều phối: `python run_simulation.py`

Script này sẽ lần lượt khởi động receiver, đợi nó sẵn sàng, khởi động sender để gửi dữ liệu qua socket, đợi sender hoàn thành (hoặc gửi đủ data), và cuối cùng khởi động tiến trình Spark training (main.py) để đọc data đã nhận và huấn luyện mô hình.

## Cấu Hình

- **Tham số huấn luyện:** Các tham số như TRAINING_EPOCHS (số vòng Spark đọc dữ liệu và tổng hợp model) và PARTITION_EPOCHS (số epoch huấn luyện cục bộ trên mỗi partition) có thể chỉnh sửa trong main.py.
- **Đường dẫn dữ liệu streaming:** Thư mục nơi receiver ghi file và Spark đọc được định nghĩa bởi STREAM_INPUT_DIR trong stream.py.
- **Cấu hình Socket và Dữ liệu gửi/nhận:** Các tham số như HOST, PORT của socket, số lượng mẫu gửi (num_samples_to_send, samples_per_send_batch trong sender_process.py) và cách receiver ghi file (samples_per_file_chunk trong receiver_process.py) có thể chỉnh sửa trong sender_process.py và receiver_process.py.
- Kiến trúc mô hình: Lớp mô hình sử dụng (SimpleCNN) được định nghĩa trong models/cnn.py.

ng tất cả các file và cấu hình cần thiết đều có sẵn cho các worker của Spark.

## Kết quả minh hoạ 
- Kết quả chạy trên máy được lưu lại trong file `PS_result.txt`.

## Ghi Chú
- Bộ dữ liệu MNIST gốc được tải xuống và chuẩn bị bởi dataloader.py, được sử dụng bởi sender_process.py.


