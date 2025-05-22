# Real-time Spam Email Detection

Hệ thống dự đoán spam email theo thời gian thực sử dụng SGDClassifier với online learning.

## Cấu trúc thư mục

```
real_time_spam_detection/
├── data/
│   └── spambase.csv               # Dữ liệu gốc từ UCI
├── model/
│   ├── spam_model.pkl             # Mô hình đã huấn luyện
│   └── scaler.pkl                 # Scaler để chuẩn hóa dữ liệu
├── app/
│   ├── streamlit_app.py           # Giao diện người dùng
│   └── stream_data_simulator.py   # Giả lập dữ liệu thời gian thực
├── train/
│   └── online_train.py            # Huấn luyện mô hình
├── requirements.txt               # Danh sách thư viện
└── README.md                      # Hướng dẫn này
```

## Cài đặt

1. **Cài đặt các thư viện cần thiết:**
```bash
pip install -r requirements.txt
```

2. **Tải dữ liệu:**
   - Tải file `spambase.data` từ [UCI Repository](https://archive.ics.uci.edu/ml/datasets/spambase)
   - Đổi tên thành `spambase.csv`
   - Đặt vào thư mục `data/`

## Hướng dẫn sử dụng

### Bước 1: Huấn luyện mô hình ban đầu

```bash
cd train/
python online_train.py
```

Lệnh này sẽ:
- Tải dữ liệu từ `data/spambase.csv`
- Huấn luyện mô hình SGDClassifier
- Lưu mô hình và scaler vào thư mục `model/`
- Hiển thị độ chính xác và báo cáo phân loại

### Bước 2: Chạy ứng dụng Streamlit

```bash
cd app/
streamlit run streamlit_app.py
```

## Tính năng

### 1. **Real-time Streaming**
- Giả lập dữ liệu email mới đến mỗi 5 giây (có thể tùy chỉnh)
- Hiển thị thông tin email: người gửi, tiêu đề, thời gian

### 2. **Dự đoán Spam**
- Sử dụng SGDClassifier để dự đoán email spam/không spam
- Hiển thị độ tin cậy và xác suất của từng lớp
- So sánh với nhãn thật để đánh giá độ chính xác

### 3. **Online Learning**
- Tự động cập nhật mô hình với dữ liệu mới
- Sử dụng `partial_fit()` của SGD để học online
- Có thể bật/tắt tính năng auto-update

### 4. **Visualizations**
- Biểu đồ độ chính xác tích lũy theo thời gian
- Thống kê real-time: tổng số email, độ chính xác
- Lịch sử 10 email gần nhất

### 5. **Giao diện Streamlit**
- Dashboard trực quan, dễ sử dụng
- Điều khiển start/stop streaming
- Cấu hình khoảng thời gian streaming
- Hiển thị kết quả real-time

## Cấu hình

### Tham số mô hình (trong `online_train.py`):
- `loss='log_loss'`: Sử dụng logistic regression
- `learning_rate='constant'`: Tốc độ học cố định
- `eta0=0.01`: Tốc độ học ban đầu

### Tham số streaming (trong `streamlit_app.py`):
- Khoảng thời gian streaming: 1-10 giây (mặc định 5 giây)
- Auto-update model: Có thể bật/tắt
- Số lượng email hiển thị trong lịch sử: 10

## Dữ liệu

Dữ liệu UCI Spambase có 58 thuộc tính:
- 57 features: tần suất từ khóa, ký tự đặc biệt, độ dài từ, v.v.
- 1 label: 0 = không spam, 1 = spam
- Tổng cộng: 4601 mẫu dữ liệu

## Lưu ý

1. **Chạy training trước:** Phải chạy `online_train.py` trước khi chạy ứng dụng
2. **Đường dẫn file:** Đảm bảo file `spambase.csv` được đặt đúng vị trí
3. **Performance:** Với dữ liệu lớn, có thể cần tối ưu streaming interval
4. **Memory:** Online learning giúp tiết kiệm bộ nhớ so với batch learning

## Mở rộng

Có thể mở rộng hệ thống với:
- Kết nối với email server thật (IMAP, POP3)
- Sử dụng mô hình phức tạp hơn (Random Forest, Neural Network)
- Thêm feature engineering (TF-IDF, Word2Vec)
- Lưu trữ dữ liệu vào database
- API REST để tích hợp với các hệ thống khác
- Cảnh báo real-time qua email/SMS
- Dashboard admin để quản lý mô hình

## Troubleshooting

### Lỗi thường gặp:

1. **"No module named 'sklearn'"**
   ```bash
   pip install scikit-learn
   ```

2. **"File not found: spambase.csv"**
   - Kiểm tra file có trong thư mục `data/`
   - Đảm bảo tên file chính xác

3. **"Cannot load model"**
   - Chạy lại `python online_train.py`
   - Kiểm tra thư mục `model/` đã được tạo

4. **Streamlit không hiển thị**
   - Kiểm tra port 8501 có bị chiếm dụng
   - Thử: `streamlit run streamlit_app.py --server.port 8502`

