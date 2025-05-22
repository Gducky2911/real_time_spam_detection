# real_time_spam_detection
Hướng dẫn chạy hệ thống:
Bước 1: Chuẩn bị dữ liệu

Tải file spambase.data từ UCI Repository
Đổi tên thành spambase.csv
Đặt vào thư mục data/

Bước 2: Cài đặt thư viện
pip install -r requirements.txt
Bước 3: Tạo thư mục cần thiết
mkdir -p model
mkdir -p .streamlit
Bước 4: Huấn luyện mô hình
cd train/
python online_train.py
Bước 5: Chạy ứng dụng
cd app/
streamlit run streamlit_app.py
