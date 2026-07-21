import os
from roboflow import Roboflow

# Đường dẫn thư mục để tải dataset về
DOWNLOAD_DIR = "datasets"

# Khởi tạo Roboflow với API Key của bạn
rf = Roboflow(api_key="B7qdNDP1W08UrhjY8J1r")

# Tải dự án lớn 10,463 ảnh (Dataset A)
# Đường dẫn/Link Roboflow Universe: https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11
project = rf.workspace("sayed-gamall").project("fire-smoke-detection-yolov11")

# Tải phiên bản mới nhất ở định dạng YOLOv11 về thư mục chỉ định
dataset = project.version(2).download("yolov11", location=DOWNLOAD_DIR)

print(f"Đã tải thành công dataset về thư mục: {os.path.abspath(dataset.location)}")
