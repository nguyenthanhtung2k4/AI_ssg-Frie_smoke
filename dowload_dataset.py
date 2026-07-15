from roboflow import Roboflow

# Khởi tạo Roboflow với API Key của bạn
rf = Roboflow(api_key="B7qdNDP1W08UrhjY8J1r")

# Tải dự án lớn 10,463 ảnh (Dataset A)
project = rf.workspace("sayed-gamall").project("fire-smoke-detection-yolov11")

# Hoặc nếu muốn tải bộ khói góc quay CCTV chéo (Dataset C):
# project = rf.workspace("iqbal-p6p0x").project("fire-smoke-cctv-angle")

# Tải phiên bản mới nhất ở định dạng YOLOv8
dataset = project.version(2).download("yolov11")

print(f"Đã tải thành công dataset về thư mục: {dataset.location}")
