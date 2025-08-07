import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Hàm tải dữ liệu từ CSV (hoặc các nguồn dữ liệu khác)
def load_data(file_path):
    """
    Tải dữ liệu từ tệp CSV.
    Args:
        file_path (str): Đường dẫn đến tệp dữ liệu.
    Returns:
        pd.DataFrame: Dữ liệu đã được tải.
    """
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        print(f"Dữ liệu đã được tải từ {file_path}")
        return data
    else:
        print(f"Tệp {file_path} không tồn tại.")
        return None

# Hàm chia tập dữ liệu thành Train/Test
def split_data(data, test_size=0.2):
    """
    Chia dữ liệu thành tập huấn luyện và kiểm tra.
    Args:
        data (pd.DataFrame): Dữ liệu.
        test_size (float): Tỉ lệ dữ liệu kiểm tra (0.2 nghĩa là 20% dữ liệu cho kiểm tra).
    Returns:
        (pd.DataFrame, pd.DataFrame): Dữ liệu huấn luyện và kiểm tra.
    """
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size)
    return train_data, test_data

# Hàm xử lý văn bản và mã hóa (tokenization)
def tokenize_text(texts, tokenizer, max_length=128):
    """
    Mã hóa văn bản bằng tokenizer.
    Args:
        texts (list of str): Danh sách các văn bản.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer đã được huấn luyện.
        max_length (int): Độ dài tối đa của chuỗi tokenized.
    Returns:
        dict: Các đầu vào đã được tokenized (input_ids, attention_mask).
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings

# Dataset class cho PyTorch
class TextDataset(Dataset):
    """
    Dataset dùng cho huấn luyện mô hình.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenize_text([text], self.tokenizer, self.max_length)
        return { 'input_ids': encoding['input_ids'][0], 
                 'attention_mask': encoding['attention_mask'][0], 
                 'label': torch.tensor(label, dtype=torch.long) }

# Hàm vẽ biểu đồ
def plot_loss_and_accuracy(losses, accuracies, filename="training_plot.png"):
    """
    Vẽ biểu đồ cho Loss và Accuracy trong quá trình huấn luyện.
    Args:
        losses (list): Danh sách các giá trị loss trong mỗi epoch.
        accuracies (list): Danh sách các giá trị accuracy trong mỗi epoch.
        filename (str): Đường dẫn tệp lưu ảnh.
    """
    plt.figure(figsize=(12, 6))
    
    # Vẽ biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Vẽ biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Biểu đồ đã được lưu tại {filename}")

# Hàm tính độ chính xác
def compute_accuracy(predictions, labels):
    """
    Tính độ chính xác giữa các dự đoán và nhãn thực.
    Args:
        predictions (torch.Tensor): Dự đoán của mô hình.
        labels (torch.Tensor): Nhãn thực tế.
    Returns:
        float: Độ chính xác của mô hình.
    """
    correct = (predictions == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy
