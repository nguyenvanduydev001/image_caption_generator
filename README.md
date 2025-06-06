# 🖼️ Image Caption Generator

Dự án sử dụng mô hình học sâu để tạo chú thích (caption) cho hình ảnh một cách tự động. Ứng dụng có giao diện đơn giản bằng Streamlit, cho phép người dùng tải ảnh lên và nhận được mô tả ngắn gọn bằng tiếng Anh.

## 📂 Dataset

Ảnh và caption được lấy từ Kaggle tại đường dẫn:

📎 [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

- Bộ dữ liệu bao gồm 8000+ hình ảnh và các chú thích tương ứng.
- Mỗi ảnh có từ 5 mô tả khác nhau, thích hợp để huấn luyện mô hình mô tả ảnh.
  
[Notebook huấn luyện mô hình ](https://www.kaggle.com/code/nguynvnduy001/image-caption-generator)

## 🛠️ Cài đặt môi trường

### Bước 1: Tải mã nguồn
```bash
git clone https://github.com/nguyenvanduydev001/image_caption_generator.git
cd image-caption-generator
```

### Bước 2: Tạo và kích hoạt môi trường ảo
```bash
# Tạo
python -m venv venv

# Kích hoạt môi trường ảo
venv\Scripts\activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

## 🚀 Chạy ứng dụng
```bash
streamlit run app.py
```

## 🧠 Kiến trúc mô hình

* **Feature extractor**: Sử dụng mô hình CNN (InceptionV3 hoặc ResNet50) để trích xuất đặc trưng từ ảnh.
* **Caption model**: Sử dụng LSTM kết hợp attention để tạo ra câu mô tả từ đặc trưng ảnh.
* **Tokenizer**: Mã hóa và giải mã chuỗi văn bản từ/đến chỉ số.

## 🎬 Demo

Người dùng chỉ cần chọn một hình ảnh → hệ thống sẽ hiển thị mô tả tự động do mô hình học sâu tạo ra.

🔗 **Demo :** [https://image-caption-genrator.streamlit.app/](https://image-caption-genrator.streamlit.app/)

![Demo Image](/demo_image.png)

## 📁 Cấu trúc thư mục

```
├── app.py                      # File chạy chính bằng Streamlit
├── models/                     # Chứa mô hình đã huấn luyện 
├── uploaded_image.jpg          # Ảnh được người dùng upload tạm thời
├── requirements.txt            # Danh sách thư viện cần cài
├── README.md                   # File mô tả dự án
└── image-caption-generator.ipynb # Notebook huấn luyện mô hình 
```

## 💼 Mục tiêu dự án

* Ứng dụng kiến thức về deep learning, NLP và xử lý ảnh.
* Triển khai mô hình trong môi trường thân thiện với người dùng.

