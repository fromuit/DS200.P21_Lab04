# models/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Kích thước ảnh sau 2 lớp pool: 28/2/2 = 7 -> 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = SimpleCNN() 
    print(model)
    # Tensor input giả (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Kiểm tra kích thước đầu vào của lớp FC
    # Tạo một tensor đi qua các lớp Conv/Pool để xem kích thước
    with torch.no_grad():
         temp_x = torch.randn(1, 1, 28, 28)
         temp_x = model.pool1(model.relu1(model.bn1(model.conv1(temp_x))))
         temp_x = model.pool2(model.relu2(model.bn2(model.conv2(temp_x))))
         temp_x = model.relu3(model.bn3(model.conv3(temp_x)))
         print(f"Shape before flattening: {temp_x.shape}")
         # Kích thước mong đợi là torch.Size([1, 128, 7, 7])
         assert temp_x.shape[1] * temp_x.shape[2] * temp_x.shape[3] == model.fc_input_dim