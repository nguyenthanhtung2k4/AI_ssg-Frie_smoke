import os
import torch
from ultralytics import YOLO

def main():
    # 1. Định nghĩa đường dẫn
    model_path = "models/best_fire_detection_v1.pt"
    data_yaml_path = "datasets/data.yaml"
    nameModel = "yolo11s_V2"
    
    # Kiểm tra sự tồn tại của file weights và file cấu hình dataset
    if not os.path.exists(model_path):
        print(f"[LỖI] Không tìm thấy file trọng số mô hình gốc tại: {model_path}")
        return
        
    if not os.path.exists(data_yaml_path):
        print(f"[LỖI] Không tìm thấy cấu hình dataset tại: {data_yaml_path}")
        return

    # 2. Khởi tạo mô hình YOLO11 với trọng số đã huấn luyện trước
    print(f"Đang tải mô hình YOLO từ: {model_path}...")
    model = YOLO(model_path)
    
    # Tự động chọn thiết bị (GPU nếu có CUDA, ngược lại dùng CPU)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị huấn luyện: {device} ({'GPU CUDA' if device == 0 else 'CPU'})")

    
    training_params = {
        "data": data_yaml_path,
        "epochs": 70,                  # Chạy 70 epoch theo yêu cầu người dùng
        "patience": 15,                # Tăng patience lên 15 để tránh dừng sớm khi đang cải thiện
        "imgsz": 640,                  # Kích thước ảnh đầu vào (khớp với mô hình gốc)
        "batch": 128,                  # Tăng batch size lên 128 tận dụng 24GB VRAM giúp train cực nhanh
        "workers": 8,                  # Tăng lên 8 workers để CPU chuẩn bị dữ liệu nhanh hơn
        "device": device,              # Thiết bị chạy (GPU/CPU)
        "optimizer": "AdamW",          # Sử dụng optimizer AdamW giúp hội tụ tốt và ổn định hơn khi fine-tune
        "lr0": 0.002,                  # Giảm học suất xuống 0.002 để fine-tune giữ đặc trưng gốc tốt hơn
        "lrf": 0.01,                   # Tỷ lệ learning rate cuối cùng
        "freeze": 10,                  # Đóng băng 10 layers backbone đầu để giữ đặc trưng cũ
        "cls": 1.5,                    # Tăng trọng số cho classification loss để tối ưu phân loại khói
        "cache": True,                 # Cache dữ liệu vào RAM (hệ thống có 126GB RAM) giúp loại bỏ nghẽn I/O ổ đĩa
        "amp": True,                   # Bật Automatic Mixed Precision (FP16) tận dụng Tensor Cores của A30
        
        # Các tham số Augmentation chuyên biệt cho camera giám sát (CCTV):
        "mosaic": 0.8,                 # Ghép ảnh nhẹ hơn (0.8) để giữ cấu trúc tự nhiên của khói
        "mixup": 0.15,                 # Trộn ảnh tăng độ tổng quát hóa
        "scale": 0.5,                  # Co giãn ảnh ngẫu nhiên
        "translate": 0.1,              # Dịch chuyển ảnh nhẹ (giả lập camera rung lắc hoặc lia góc)
        "fliplr": 0.5,                 # Lật ảnh ngang ngẫu nhiên
        
        # Giả lập thay đổi ánh sáng thời tiết trên CCTV:
        "hsv_h": 0.015,                # Điều chỉnh tông màu nhẹ
        "hsv_s": 0.7,                  # Điều chỉnh độ bão hòa màu sắc (mô phỏng cam CCTV chất lượng thấp)
        "hsv_v": 0.4,                  # Điều chỉnh độ sáng tối (mô phỏng chuyển đổi ngày/đêm)
        
        # Cấu hình lưu trữ
        # "project": "runs",     # Thư mục lưu kết quả huấn luyện (sẽ lưu dưới runs/nameModel)
        "name": nameModel,             # Tên tiến trình train
        "exist_ok": True,              # Cho phép ghi đè/lưu đè nếu chạy lại tiến trình cùng tên
        "plots": True                  # Vẽ biểu đồ loss, metrics để theo dõi
    }
    
    print("\n" + "="*50)
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN TIẾP TỤC (FINE-TUNING)")
    print("="*50)
    for k, v in training_params.items():
        print(f" - {k:<15}: {v}")
    print("="*50 + "\n")
    
    # 4. Thực thi huấn luyện
    model.train(**training_params)
    
    print("\nQuá trình huấn luyện hoàn tất!")
    print(f"Trọng số mô hình tốt nhất được lưu tại: runs/{nameModel}")

if __name__ == "__main__":
    main()
