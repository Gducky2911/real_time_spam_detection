import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data():
    """Tải dữ liệu spam từ file CSV"""
    data_path = '../data/spambase.csv'
    
    # Đọc dữ liệu
    data = pd.read_csv(data_path, header=None)
    
    # Tách features và labels
    X = data.iloc[:, :-1].values  # 57 features đầu
    y = data.iloc[:, -1].values   # Cột cuối là label (spam/not spam)
    
    print(f"Dữ liệu có {X.shape[0]} mẫu với {X.shape[1]} đặc trưng")
    print(f"Số lượng spam: {np.sum(y == 1)}, Số lượng không spam: {np.sum(y == 0)}")
    
    return X, y

def train_online_model():
    """Huấn luyện mô hình SGD với online learning"""
    # Tải dữ liệu
    X, y = load_data()
    
    # Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Khởi tạo mô hình SGD cho online learning
    model = SGDClassifier(
        loss='log_loss',           # Sử dụng logistic regression
        learning_rate='constant',   # Tốc độ học cố định
        eta0=0.01,                 # Tốc độ học ban đầu
        random_state=42,
        max_iter=1000
    )
    
    # Huấn luyện ban đầu
    print("Đang huấn luyện mô hình...")
    model.fit(X_train_scaled, y_train)
    
    # Đánh giá mô hình
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Độ chính xác trên tập test: {accuracy:.4f}")
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))
    
    # Lưu mô hình và scaler
    model_dir = '../model'
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, f'{model_dir}/spam_model.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    
    print(f"Mô hình đã được lưu vào {model_dir}/spam_model.pkl")
    print(f"Scaler đã được lưu vào {model_dir}/scaler.pkl")
    
    return model, scaler

def update_model_online(model, scaler, new_X, new_y):
    """Cập nhật mô hình với dữ liệu mới (online learning)"""
    # Chuẩn hóa dữ liệu mới
    new_X_scaled = scaler.transform(new_X.reshape(1, -1))
    
    # Cập nhật mô hình với dữ liệu mới
    model.partial_fit(new_X_scaled, [new_y])
    
    return model

if __name__ == "__main__":
    # Huấn luyện mô hình ban đầu
    model, scaler = train_online_model()