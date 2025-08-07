import pandas as pd
from sklearn.model_selection import train_test_split
from underthesea import word_tokenize
from transformers import AutoTokenizer

# Hàm đọc dữ liệu từ CSV
def load_data(file_path):
    """
    Đọc dữ liệu từ tệp CSV vào DataFrame
    Args:
        file_path (str): Đường dẫn tệp dữ liệu
    Returns:
        pd.DataFrame: Dữ liệu dưới dạng DataFrame
    """
    data = pd.read_csv(file_path)
    return data

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    """
    Tiền xử lý văn bản: loại bỏ ký tự đặc biệt, chuyển sang chữ thường và tokenize.
    Args:
        text (str): Văn bản cần xử lý
    Returns:
        str: Văn bản đã được tiền xử lý
    """
    # Chuyển văn bản thành chữ thường và loại bỏ ký tự không cần thiết
    text = text.lower()
    text = " ".join(word_tokenize(text))  # Sử dụng word_tokenize từ thư viện underthesea để tokenize
    return text

# Hàm chuẩn bị dữ liệu cho mô hình
def prepare_data(data, text_column, label_column, test_size=0.2):
    """
    Chuẩn bị dữ liệu: Tiền xử lý văn bản và chia dữ liệu thành Train/Test
    Args:
        data (pd.DataFrame): Dữ liệu
        text_column (str): Tên cột chứa văn bản
        label_column (str): Tên cột chứa nhãn
        test_size (float): Tỉ lệ dữ liệu kiểm tra (default là 0.2)
    Returns:
        pd.DataFrame: Dữ liệu đã được chia thành train và test
    """
    # Tiền xử lý văn bản
    data[text_column] = data[text_column].apply(preprocess_text)

    # Chia dữ liệu thành train và test
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    return train_data, test_data

# Hàm tạo DataLoader cho PyTorch
def create_dataloader(data, tokenizer, max_length=128, batch_size=16):
    """
    Tạo DataLoader cho mô hình PyTorch
    Args:
        data (pd.DataFrame): Dữ liệu đã được tiền xử lý
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer đã được huấn luyện
        max_length (int): Độ dài tối đa của chuỗi tokenized
        batch_size (int): Kích thước batch
    Returns:
        torch.utils.data.DataLoader: DataLoader cho PyTorch
    """
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    import torch

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return { 'input_ids': encoding['input_ids'].flatten(), 
                     'attention_mask': encoding['attention_mask'].flatten(), 
                     'labels': torch.tensor(label, dtype=torch.long) }

    dataset = TextDataset(
        texts=data['text'].tolist(),
        labels=data['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Hàm tải tokenizer
def load_tokenizer(model_name='vinai/phobert-base'):
    """
    Tải tokenizer từ transformers
    Args:
        model_name (str): Tên mô hình để tải tokenizer (ví dụ: 'vinai/phobert-base')
    Returns:
        transformers.PreTrainedTokenizer: Tokenizer đã được tải
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
