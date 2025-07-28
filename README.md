# Underwater YoloV8 Object Detection

<div align="center">
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Computer_Vision-Deep_Learning-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Underwater-Object_Detection-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Dataset-Aquarium_COTS-red?style=for-the-badge">
</div>

## 📋 Tổng Quan (Overview)

Dự án này triển khai mô hình YOLOv8 để phát hiện và nhận diện các sinh vật biển trong môi trường dưới nước. Sử dụng tập dữ liệu Aquarium COTS với 7 lớp đối tượng khác nhau, mô hình có thể nhận diện chính xác các loài sinh vật biển từ hình ảnh.

This project implements YOLOv8 for detecting and recognizing marine creatures in underwater environments. Using the Aquarium COTS dataset with 7 different object classes, the model can accurately identify marine species from images.

## 🎯 Các Lớp Đối Tượng (Object Classes)

Mô hình có thể phát hiện 7 loại sinh vật biển sau:

| ID | Tên Lớp | Mô Tả |
|----|----------|-------|
| 0 | Fish | Các loài cá |
| 1 | Jellyfish | Sứa |
| 2 | Penguin | Chim cánh cụt |
| 3 | Puffin | Chim hải âu |
| 4 | Shark | Cá mập |
| 5 | Starfish | Sao biển |
| 6 | Stingray | Cá đuối |

## 🚀 Tính Năng Chính (Key Features)

- **Hiệu Suất Cao**: Sử dụng YOLOv8 architecture cho tốc độ inference nhanh
- **Đa Dạng Sinh Vật**: Phát hiện 7 loài sinh vật biển khác nhau
- **Môi Trường Dưới Nước**: Tối ưu hóa cho điều kiện ánh sáng và độ trong suốt đặc biệt
- **Training Pipeline**: Pipeline huấn luyện hoàn chỉnh với configuration linh hoạt
- **GPU Support**: Hỗ trợ training trên Tesla T4 GPU

## 🛠️ Công Nghệ Sử Dụng (Technology Stack)

- **Framework**: Ultralytics YOLOv8 (v8.1.37)
- **Deep Learning**: PyTorch
- **Computer Vision**: OpenCV, PIL
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Environment**: Kaggle Notebook, Tesla T4 GPU

## 📊 Cấu Trúc Dữ Liệu (Dataset Structure)

```
aquarium_pretrain/
├── train/
│   ├── images/          # Hình ảnh training
│   └── labels/          # Annotations YOLO format
├── valid/
│   ├── images/          # Hình ảnh validation  
│   └── labels/          # Annotations validation
└── test/
    ├── images/          # Hình ảnh test
    └── labels/          # Annotations test
```

### Định Dạng Annotation
- **Format**: YOLO (.txt files)
- **Structure**: `[class_id, center_x, center_y, width, height]`
- **Coordinates**: Normalized (0-1 range)

## ⚙️ Cấu Hình Training (Training Configuration)

```python
# Model Configuration
BASE_MODEL = 'yolov8n'        # Nano version for fast inference
EPOCHS = 100                  # Training epochs
BATCH_SIZE = 32              # Batch size
OPTIMIZER = 'auto'           # Automatic optimizer selection
LEARNING_RATE = 0.001        # Initial learning rate
WEIGHT_DECAY = 5e-4          # L2 regularization
PATIENCE = 20                # Early stopping patience
```

## 🔧 Cài Đặt và Sử Dụng (Installation & Usage)

### 1. Cài Đặt Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib seaborn
pip install torch torchvision
```

### 2. Chuẩn Bị Dữ Liệu

```python
# Create YAML configuration file
dict_file = {
    'train': 'path/to/train',
    'val': 'path/to/valid', 
    'test': 'path/to/test',
    'nc': 7,
    'names': ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
}
```

### 3. Training Model

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    batch=32,
    patience=20,
    optimizer='auto'
)
```

### 4. Inference

```python
# Load trained model
model = YOLO('best.pt')

# Predict on new images
results = model('underwater_image.jpg')
results[0].show()
```

## 📈 Hiệu Suất Mô Hình (Model Performance)

- **Architecture**: YOLOv8 Nano
- **Training Time**: ~3 hours trên Tesla T4 GPU
- **Image Resolution**: 1024x768 pixels
- **Inference Speed**: Real-time detection capability
- **Hardware**: Tesla T4 GPU (dual GPU setup)

## 🖼️ Ví Dụ Kết Quả (Sample Results)

Mô hình có thể phát hiện và nhận diện chính xác các sinh vật biển trong nhiều điều kiện khác nhau:

- Môi trường nước trong
- Ánh sáng yếu dưới nước  
- Nhiều đối tượng trong cùng một khung hình
- Các loài có kích thước và hình dạng đa dạng

## 🔄 Pipeline Xử Lý (Processing Pipeline)

1. **Data Loading**: Load và preprocess hình ảnh training
2. **Augmentation**: Áp dụng data augmentation tự động
3. **Model Training**: Training với early stopping
4. **Validation**: Đánh giá trên tập validation
5. **Testing**: Test trên tập dữ liệu test riêng biệt
6. **Inference**: Deployment cho prediction thời gian thực

## 📋 Yêu Cầu Hệ Thống (System Requirements)

### Minimum Requirements:
- **RAM**: 8GB+
- **GPU**: GTX 1060 hoặc tương đương
- **Storage**: 5GB free space
- **Python**: 3.8+

### Recommended:
- **RAM**: 16GB+  
- **GPU**: RTX 3080 hoặc Tesla T4
- **Storage**: SSD với 10GB+ free space
- **Python**: 3.9+

## 📚 Tài Liệu Tham Khảo (References)

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Aquarium Dataset](https://www.kaggle.com/datasets/solesensei/aquarium-data-cots)
- [YOLO Format Annotation Guide](https://docs.ultralytics.com/datasets/detect/)
- [Underwater Computer Vision](https://paperswithcode.com/task/underwater-object-detection)

## 🤝 Đóng Góp (Contributing)

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Dự án này được phân phối dưới MIT License. Xem file `LICENSE` để biết thêm chi tiết.

## 📞 Liên Hệ (Contact)

- **Author**: kenzn2
- **GitHub**: [@kenzn2](https://github.com/kenzn2)
- **Project**: [underwater-YoloV8](https://github.com/kenzn2/underwater-YoloV8)

## 🙏 Acknowledgments

- Ultralytics team cho YOLOv8 framework
- Kaggle community cho Aquarium COTS dataset
- Contributors và maintainers của các open-source libraries

---

<div align="center">
  <b>🌊 Khám phá thế giới dưới nước với AI! 🐠</b>
</div>