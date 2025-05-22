import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import threading
import queue

class SpamDataStreamer:
    def __init__(self, data_path='../data/spambase.csv'):
        """Khởi tạo streamer với dữ liệu spam"""
        self.data_path = data_path
        self.data = None
        self.current_index = 0
        self.is_streaming = False
        self.data_queue = queue.Queue()
        self.load_data()
        
    def load_data(self):
        """Tải dữ liệu spam"""
        try:
            self.data = pd.read_csv(self.data_path, header=None)
            print(f"Đã tải {len(self.data)} mẫu dữ liệu")
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            
    def shuffle_data(self):
        """Trộn dữ liệu để giả lập luồng ngẫu nhiên"""
        if self.data is not None:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
            self.current_index = 0
            
    def get_next_sample(self):
        """Lấy mẫu dữ liệu tiếp theo"""
        if self.data is None or self.current_index >= len(self.data):
            return None
            
        # Lấy một dòng dữ liệu
        sample = self.data.iloc[self.current_index]
        
        # Tạo thông tin email giả lập
        email_info = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'email_id': f"email_{self.current_index + 1:05d}",
            'features': sample.iloc[:-1].values,  # 57 features
            'true_label': int(sample.iloc[-1]),   # Label thật
            'sender': self._generate_fake_sender(),
            'subject': self._generate_fake_subject(int(sample.iloc[-1]))
        }
        
        self.current_index += 1
        return email_info
        
    def _generate_fake_sender(self):
        """Tạo địa chỉ email người gửi giả"""
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
                  'company.com', 'spammer.net', 'promo.biz']
        names = ['john', 'mary', 'admin', 'support', 'noreply', 'promo', 
                'offer', 'winner', 'lucky', 'deal']
        
        name = random.choice(names)
        domain = random.choice(domains)
        return f"{name}@{domain}"
        
    def _generate_fake_subject(self, is_spam):
        """Tạo tiêu đề email giả dựa trên label"""
        if is_spam:
            spam_subjects = [
                "🎉 CONGRATULATIONS! You've won $1000!",
                "URGENT: Claim your prize now!",
                "💰 Make money fast - Work from home",
                "Free iPhone - Click here now!",
                "SALE: 90% OFF Everything!",
                "Your account will be suspended",
                "Prescription drugs - No doctor needed"
            ]
            return random.choice(spam_subjects)
        else:
            normal_subjects = [
                "Meeting reminder for tomorrow",
                "Project update - Q1 Report",
                "Invoice #12345 attached",
                "Welcome to our newsletter",
                "Your order has been shipped",
                "Weekly team standup notes",
                "Conference registration confirmed"
            ]
            return random.choice(normal_subjects)
            
    def start_streaming(self, interval=5):
        """Bắt đầu streaming dữ liệu với khoảng thời gian cho trước"""
        self.is_streaming = True
        self.shuffle_data()
        
        def stream_worker():
            while self.is_streaming:
                sample = self.get_next_sample()
                if sample is None:
                    # Hết dữ liệu, trộn lại và bắt đầu lại
                    self.shuffle_data()
                    continue
                    
                self.data_queue.put(sample)
                time.sleep(interval)
                
        self.stream_thread = threading.Thread(target=stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
    def stop_streaming(self):
        """Dừng streaming"""
        self.is_streaming = False
        
    def get_latest_sample(self):
        """Lấy mẫu mới nhất từ queue"""
        try:
            return self.data_queue.get(timeout=1)
        except queue.Empty:
            return None
            
    def has_new_data(self):
        """Kiểm tra có dữ liệu mới không"""
        return not self.data_queue.empty()

# Test streaming
if __name__ == "__main__":
    print("Testing SpamDataStreamer...")
    
    streamer = SpamDataStreamer()
    streamer.start_streaming(interval=2)  # Stream mỗi 2 giây
    
    print("Streaming started. Press Ctrl+C to stop...")
    
    try:
        while True:
            if streamer.has_new_data():
                sample = streamer.get_latest_sample()
                if sample:
                    label_text = "SPAM" if sample['true_label'] == 1 else "NOT SPAM"
                    print(f"\n[{sample['timestamp']}] New email:")
                    print(f"  ID: {sample['email_id']}")
                    print(f"  From: {sample['sender']}")
                    print(f"  Subject: {sample['subject']}")
                    print(f"  True Label: {label_text}")
                    
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping streamer...")
        streamer.stop_streaming()
        print("Streamer stopped.")