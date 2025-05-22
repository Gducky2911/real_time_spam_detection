import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Thêm đường dẫn để import module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stream_data_simulator import SpamDataStreamer

# Cấu hình trang
st.set_page_config(
    page_title="Real-time Spam Email Detection",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Tải mô hình đã huấn luyện"""
    try:
        model = joblib.load('../model/spam_model.pkl')
        scaler = joblib.load('../model/scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Không thể tải mô hình: {e}")
        return None, None

@st.cache_resource
def initialize_streamer():
    """Khởi tạo data streamer"""
    return SpamDataStreamer('../data/spambase.csv')

def predict_email(model, scaler, features):
    """Dự đoán email spam"""
    try:
        # Chuẩn hóa features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Dự đoán
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        return None, None

def update_model_online(model, scaler, features, true_label):
    """Cập nhật mô hình với dữ liệu mới"""
    try:
        features_scaled = scaler.transform(features.reshape(1, -1))
        model.partial_fit(features_scaled, [true_label])
        return True
    except Exception as e:
        st.error(f"Lỗi khi cập nhật mô hình: {e}")
        return False

def main():
    # Tiêu đề
    st.title("🚨 Real-time Spam Email Detection")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Cấu hình")
    
    # Tải mô hình
    model, scaler = load_model()
    if model is None or scaler is None:
        st.error("Vui lòng chạy train/online_train.py trước để tạo mô hình!")
        return
    
    # Khởi tạo streamer
    if 'streamer' not in st.session_state:
        st.session_state.streamer = initialize_streamer()
        st.session_state.streaming = False
        st.session_state.email_history = []
        st.session_state.predictions = []
        st.session_state.true_labels = []
        st.session_state.timestamps = []
        st.session_state.total_emails = 0
        st.session_state.correct_predictions = 0
    
    # Điều khiển streaming
    col1, col2 = st.sidebar.columns(2)
    
    stream_interval = st.sidebar.slider("Khoảng thời gian (giây)", 1, 10, 5)
    auto_update = st.sidebar.checkbox("Tự động cập nhật mô hình", value=True)
    
    with col1:
        if st.button("▶️ Bắt đầu"):
            if not st.session_state.streaming:
                st.session_state.streamer.start_streaming(stream_interval)
                st.session_state.streaming = True
                st.success("Đã bắt đầu streaming!")
    
    with col2:
        if st.button("⏹️ Dừng"):
            if st.session_state.streaming:
                st.session_state.streamer.stop_streaming()
                st.session_state.streaming = False
                st.info("Đã dừng streaming!")
    
    # Thống kê
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Thống kê")
    st.sidebar.metric("Tổng số email", st.session_state.total_emails)
    
    if st.session_state.total_emails > 0:
        accuracy = st.session_state.correct_predictions / st.session_state.total_emails
        st.sidebar.metric("Độ chính xác", f"{accuracy:.2%}")
    
    # Layout chính
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📧 Email mới nhất")
        email_container = st.container()
        
    with col2:
        st.header("🎯 Kết quả dự đoán")
        prediction_container = st.container()
    
    # Biểu đồ
    st.header("📈 Thống kê theo thời gian")
    chart_container = st.container()
    
    # Lịch sử email
    st.header("📝 Lịch sử email")
    history_container = st.container()
    
    # Auto-refresh
    placeholder = st.empty()
    
    while True:
        if st.session_state.streaming and st.session_state.streamer.has_new_data():
            # Lấy email mới
            email_data = st.session_state.streamer.get_latest_sample()
            
            if email_data:
                # Dự đoán
                prediction, probability = predict_email(model, scaler, email_data['features'])
                
                if prediction is not None:
                    # Cập nhật session state
                    st.session_state.total_emails += 1
                    st.session_state.predictions.append(prediction)
                    st.session_state.true_labels.append(email_data['true_label'])
                    st.session_state.timestamps.append(datetime.now())
                    
                    # Kiểm tra độ chính xác
                    if prediction == email_data['true_label']:
                        st.session_state.correct_predictions += 1
                    
                    # Thêm vào lịch sử
                    email_data['prediction'] = prediction
                    email_data['probability'] = probability
                    st.session_state.email_history.append(email_data)
                    
                    # Cập nhật mô hình (online learning)
                    if auto_update:
                        update_model_online(model, scaler, email_data['features'], email_data['true_label'])
                    
                    # Hiển thị email mới nhất
                    with email_container:
                        st.markdown(f"**⏰ Thời gian:** {email_data['timestamp']}")
                        st.markdown(f"**📧 ID:** {email_data['email_id']}")
                        st.markdown(f"**👤 Từ:** {email_data['sender']}")
                        st.markdown(f"**📋 Tiêu đề:** {email_data['subject']}")
                        
                        # Hiển thị nhãn thật
                        true_label_text = "🔴 SPAM" if email_data['true_label'] == 1 else "✅ NOT SPAM"
                        st.markdown(f"**🏷️ Nhãn thật:** {true_label_text}")
                    
                    # Hiển thị dự đoán
                    with prediction_container:
                        pred_text = "🔴 SPAM" if prediction == 1 else "✅ NOT SPAM"
                        confidence = max(probability) * 100
                        
                        if prediction == 1:
                            st.error(f"**Dự đoán:** {pred_text}")
                            st.error(f"**Độ tin cậy:** {confidence:.1f}%")
                        else:
                            st.success(f"**Dự đoán:** {pred_text}")
                            st.success(f"**Độ tin cậy:** {confidence:.1f}%")
                        
                        # Hiển thị xác suất
                        st.write("**Xác suất:**")
                        st.write(f"- Not Spam: {probability[0]:.3f}")
                        st.write(f"- Spam: {probability[1]:.3f}")
                        
                        # Kiểm tra dự đoán đúng/sai
                        if prediction == email_data['true_label']:
                            st.success("✅ Dự đoán chính xác!")
                        else:
                            st.error("❌ Dự đoán sai!")
        
        # Cập nhật biểu đồ
        if len(st.session_state.timestamps) > 0:
            with chart_container:
                # Biểu đồ độ chính xác theo thời gian
                df_chart = pd.DataFrame({
                    'Timestamp': st.session_state.timestamps,
                    'Prediction': st.session_state.predictions,
                    'True_Label': st.session_state.true_labels
                })
                
                # Tính độ chính xác tích lũy
                df_chart['Correct'] = (df_chart['Prediction'] == df_chart['True_Label']).astype(int)
                df_chart['Cumulative_Accuracy'] = df_chart['Correct'].expanding().mean()
                
                fig = px.line(df_chart, x='Timestamp', y='Cumulative_Accuracy', 
                             title='Độ chính xác tích lũy theo thời gian')
                fig.update_yaxis(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị lịch sử
        if st.session_state.email_history:
            with history_container:
                # Chỉ hiển thị 10 email gần nhất
                recent_emails = st.session_state.email_history[-10:]
                
                history_data = []
                for email in reversed(recent_emails):
                    pred_text = "SPAM" if email['prediction'] == 1 else "NOT SPAM"
                    true_text = "SPAM" if email['true_label'] == 1 else "NOT SPAM"
                    correct = "✅" if email['prediction'] == email['true_label'] else "❌"
                    
                    history_data.append({
                        'Thời gian': email['timestamp'],
                        'ID': email['email_id'],
                        'Người gửi': email['sender'],
                        'Tiêu đề': email['subject'][:50] + "..." if len(email['subject']) > 50 else email['subject'],
                        'Dự đoán': pred_text,
                        'Thực tế': true_text,
                        'Đúng/Sai': correct
                    })
                
                df_history = pd.DataFrame(history_data)
                st.dataframe(df_history, use_container_width=True)
        
        # Chờ 1 giây trước khi refresh
        time.sleep(1)
        
        # Rerun để cập nhật giao diện
        if st.session_state.streaming:
            st.rerun()

if __name__ == "__main__":
    main()