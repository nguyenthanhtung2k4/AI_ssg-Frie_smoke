import os
import sys
import time
import logging
import configparser
import cv2
import torch
from ultralytics import YOLO

class FireSmokeDetectionEngine:
    def __init__(self, config_path="config.txt"):
        self.config_path = config_path
        self.model = None
        self.cap = None
        self.writer = None
        self.window_name = "He Thong Phat Hien Lua & Khoi AI"
        
        # 1. Tải cấu hình từ file
        self.load_config()
        
        # Tự động kiểm tra và chọn thiết bị chạy (Sử dụng GPU CUDA nếu có, ngược lại dùng CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 2. Thiết lập thư mục và công cụ ghi log (Logger)
        self.logger, self.log_file = self.setup_logger("logs")
        self.logger.info("=" * 60)
        self.logger.info("KHỞI TẠO TIẾN TRÌNH NHẬN DIỆN LỬA & KHÓI AI")
        self.logger.info("=" * 60)
        self.logger.info(f"Đã tải cấu hình từ: {self.config_path}")
        
        # Ghi log chi tiết các thông số cấu hình khởi chạy
        self.logger.info(f"Đường dẫn mô hình (Model): {self.model_path}")
        self.logger.info(f"Ngưỡng tin cậy chung (Confidence): {self.conf_threshold}")
        self.logger.info(f"Ngưỡng tin cậy Lửa (Fire Conf): {self.conf_threshold_fire}")
        self.logger.info(f"Ngưỡng tin cậy Khói (Smoke Conf): {self.conf_threshold_smoke}")
        self.logger.info(f"Ngưỡng đè khít (IoU): {self.iou_threshold}")
        self.logger.info(f"Thiết bị tính toán (Tự động): {self.device.upper()}")
        self.logger.info(f"Chế độ chạy: {self.mode} ({'1 - Xử lý Video Lưu File' if self.mode == 1 else '2 - Phát luồng Real-Time'})")
        self.logger.info(f"Nguồn đầu vào (Source): {self.source}")
        self.logger.info(f"Đường dẫn video đầu ra: {self.output_path}")
        self.logger.info(f"Lưu định dạng H.264 (avc1): {self.save_h264}")
        self.logger.info(f"Hiển thị màn hình GUI: {self.show_display}")
        self.logger.info(f"Độ dày Box: {self.box_thickness} | Kích thước chữ: {self.text_scale} | Độ dài góc: {self.corner_length}")
        self.logger.info(f"Tự động dò tìm SSH/Headless: {self.auto_headless_detect}")
        self.logger.info(f"Bật/Tắt ghi file log: {self.save_log}")
        self.logger.info(f"Bật/Tắt tính năng cảnh báo (Alerts): {self.enable_alerts}")
        self.logger.info(f"Số frames kích hoạt cảnh báo: {self.alert_frames_threshold}")
        self.logger.info(f"Chế độ lưu ảnh cảnh báo: {self.save_alert_images}")
        self.logger.info(f"Thời gian chờ chụp ảnh cảnh báo (Frames): {self.alert_save_cooldown_frames}")
        
        # Khởi tạo thư mục gốc chứa ảnh chụp cảnh báo
        os.makedirs("alerts", exist_ok=True)
        
        # Các biến theo dõi hiệu năng và kết quả
        self.total_frames = 0
        self.processed_frames = 0
        self.total_inference_time = 0.0
        self.total_fps_time = 0.0
        self.max_fire_count = 0
        self.max_smoke_count = 0
        
        # Các biến điều khiển cảnh báo
        self.consecutive_detections = 0
        self.alert_triggered = False
        self.last_alert_save_frame = -9999
        self.alert_event_active = False  # Theo dõi xem trạng thái cảnh báo đang hoạt động hay không
        
        # Tự động kiểm tra môi trường chạy để tắt hiển thị nếu chạy qua SSH / Headless
        self.show_display = self.check_gui_available()

    def load_config(self):
        # Nếu file cấu hình không tồn tại, tự động tạo file config.txt mặc định
        if not os.path.exists(self.config_path):
            print(f"[CẢNH BÁO] Không tìm thấy file cấu hình '{self.config_path}'. Đang tạo file config.txt mặc định...")
            self.create_default_config()
            
        config = configparser.ConfigParser()
        config.read(self.config_path, encoding='utf-8')
        
        # Phần MODEL
        self.model_path = config.get('MODEL', 'model_path', fallback='models/best_fire_detection_v1.pt')
        self.conf_threshold = config.getfloat('MODEL', 'conf_threshold', fallback=0.40)
        self.conf_threshold_fire = config.getfloat('MODEL', 'conf_threshold_fire', fallback=max(self.conf_threshold, 0.50))
        self.conf_threshold_smoke = config.getfloat('MODEL', 'conf_threshold_smoke', fallback=max(self.conf_threshold, 0.55))
        self.iou_threshold = config.getfloat('MODEL', 'iou_threshold', fallback=0.45)
        
        # Phần INFERENCE
        self.mode = config.getint('INFERENCE', 'mode', fallback=1)
        self.source = config.get('INFERENCE', 'source', fallback='demo/fire_test3.mp4')
        self.output_path = config.get('INFERENCE', 'output_path', fallback='demo/output_fire_detection.mp4')
        self.save_h264 = config.getboolean('INFERENCE', 'save_h264', fallback=True)
        
        # Phần VISUALIZATION
        self.show_display = config.getboolean('VISUALIZATION', 'show_display', fallback=True)
        self.auto_headless_detect = config.getboolean('VISUALIZATION', 'auto_headless_detect', fallback=True)
        self.box_thickness = config.getint('VISUALIZATION', 'box_thickness', fallback=3)
        self.text_scale = config.getfloat('VISUALIZATION', 'text_scale', fallback=0.5)
        self.corner_length = config.getint('VISUALIZATION', 'corner_length', fallback=15)
        
        # Phần LOGS
        self.save_log = config.getboolean('LOGS', 'save_log', fallback=True)
        
        # Phần ALERTS
        self.enable_alerts = config.getboolean('ALERTS', 'enable_alerts', fallback=True)
        self.alert_frames_threshold = config.getint('ALERTS', 'alert_frames_threshold', fallback=5)
        self.save_alert_images = config.getboolean('ALERTS', 'save_alert_images', fallback=True)
        self.alert_save_cooldown_frames = config.getint('ALERTS', 'alert_save_cooldown_frames', fallback=30)

    def create_default_config(self):
        default_content = """# Cấu hình Hệ thống Phát hiện Hỏa hoạn & Khói bằng AI

[MODEL]
model_path = models/best_fire_detection_v1.pt
conf_threshold = 0.40
conf_threshold_fire = 0.50
conf_threshold_smoke = 0.55
iou_threshold = 0.45

[INFERENCE]
mode = 1
source = demo/fire_test3.mp4
output_path = demo/output_fire_detection.mp4
save_h264 = True

[VISUALIZATION]
show_display = True
auto_headless_detect = True
box_thickness = 3
text_scale = 0.5
corner_length = 15

[LOGS]
save_log = True

[ALERTS]
enable_alerts = True
alert_frames_threshold = 5
save_alert_images = True
alert_save_cooldown_frames = 30
"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(default_content)

    def setup_logger(self, log_dir):
        logger = logging.getLogger("FireSmokeDetection")
        logger.setLevel(logging.DEBUG)
        
        # Xóa các handler cũ để tránh ghi trùng lặp log khi khởi chạy lại
        if logger.handlers:
            logger.handlers.clear()
            
        log_file = None
        # Chỉ khởi tạo ghi file log nếu cấu hình save_log là True
        if self.save_log:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"inference_{timestamp}.log")
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        # Console handler (Luôn in ra màn hình console)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger, log_file

    def check_gui_available(self):
        if not self.show_display:
            return False
            
        if self.auto_headless_detect:
            # Phát hiện nếu đang chạy qua phiên SSH (Cả Windows và Linux)
            ssh_vars = ['SSH_CLIENT', 'SSH_TTY', 'SSH_CONNECTION', 'SSH_AUTH_SOCK']
            if any(var in os.environ for var in ssh_vars):
                self.logger.warning("Phát hiện môi trường kết nối SSH. Tự động chuyển sang chế độ không màn hình (Headless) để ngăn lỗi GUI.")
                return False
                
            # Kiểm tra biến DISPLAY trên Linux
            if sys.platform != 'win32' and 'DISPLAY' not in os.environ:
                self.logger.warning("Không phát hiện biến DISPLAY trên Linux. Tắt chế độ hiển thị cửa sổ hình ảnh.")
                return False
                
        return True

    def load_model(self):
        self.logger.info(f"Đang tải mô hình phát hiện hỏa hoạn từ: {self.model_path}...")
        if not os.path.exists(self.model_path):
            self.logger.error(f"Tệp tin mô hình phát hiện không tồn tại: {self.model_path}")
            sys.exit(1)
        
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Đã nạp mô hình thành công. Thiết bị sử dụng: {self.device.upper()}")
        except Exception as e:
            self.logger.error(f"Gặp sự cố khi nạp mô hình YOLO: {e}")
            sys.exit(1)

    def draw_futuristic_box(self, img, bbox, label, color, thickness=3, corner_len=15):
        # Hàm vẽ bounding box viền góc phong cách công nghệ / giám sát hiện đại
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        t_corner = thickness + 2
        # Góc Trên - Trái
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, t_corner)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, t_corner)
        # Góc Trên - Phải
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, t_corner)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, t_corner)
        # Góc Dưới - Trái
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, t_corner)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, t_corner)
        # Góc Dưới - Phải
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t_corner)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, t_corner)
        
        # Thiết lập độ dày nét chữ dựa trên độ dày viền hộp
        font_thickness = max(1, int(thickness - 1))
        
        # Vẽ nhãn thông tin lớp đối tượng và độ tin cậy
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, font_thickness)
        cv2.rectangle(img, (x1, y1 - h - 6), (x1 + w + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def draw_hud(self, frame, fire_count, smoke_count, inference_time, current_fps):
        height, width = frame.shape[:2]
        
        # Tạo một khung đen mờ làm HUD điều khiển ở góc trên màn hình
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        
        # 1. Vẽ Badge thông báo trạng thái hoạt động (dùng tiếng Việt không dấu cho OpenCV)
        if self.alert_triggered:
            blink = int(time.time() * 4) % 2
            badge_color = (0, 0, 255) if blink == 1 else (0, 0, 160)
            cv2.rectangle(frame, (10, 10), (210, 40), badge_color, -1)
            cv2.putText(frame, "CANH BAO NGUY HIEM", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (10, 10), (105, 40), (0, 150, 0), -1)
            cv2.putText(frame, "AN TOAN", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
        # 2. Hiển thị thông số đếm đối tượng trong frame hiện tại
        fire_col = (0, 0, 255) if fire_count > 0 else (200, 200, 200)
        smoke_col = (0, 140, 255) if smoke_count > 0 else (200, 200, 200)
        
        cv2.putText(frame, f"Lua (Fire): {fire_count}", (240, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fire_col, 2 if fire_count > 0 else 1, cv2.LINE_AA)
        cv2.putText(frame, f"Khoi (Smoke): {smoke_count}", (390, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, smoke_col, 2 if smoke_count > 0 else 1, cv2.LINE_AA)
        
        # 3. Hiển thị thông số FPS và Latency mô hình xử lý
        fps_text = f"FPS: {current_fps:.1f} (Inf: {inference_time:.1f}ms)"
        cv2.putText(frame, fps_text, (width - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def save_alert_screenshot(self, frame, fire_count, smoke_count):
        # Lấy ngày hiện tại làm thư mục con: alerts/YYYY-MM-DD
        date_folder = time.strftime("%Y-%m-%d")
        target_dir = os.path.join("alerts", date_folder)
        os.makedirs(target_dir, exist_ok=True)
        
        # Định dạng ngày tháng giờ phút giây: DDMM_HHMMSS
        suffix = time.strftime("%d%m_%H%M%S")
        filename = os.path.join(target_dir, f"alert_frame_{self.processed_frames}_{suffix}.jpg")
        
        cv2.imwrite(filename, frame)
        abs_path = os.path.abspath(filename)
        self.logger.warning(f"📸 [CHỤP CẢNH BÁO] Đã lưu ảnh chụp sự cố tại Frame {self.processed_frames}: {abs_path} (Lửa: {fire_count}, Khói: {smoke_count})")

    def run(self):
        source = self.source
        is_webcam = False
        
        # Phân tích nguồn đầu vào là camera hay là video
        if str(source).isdigit():
            source = int(source)
            is_webcam = True
            self.logger.info(f"Kết nối tới máy quay (Webcam) chỉ số: {source}...")
        elif str(source).startswith("rtsp://") or str(source).startswith("rtmp://") or str(source).startswith("http://") or str(source).startswith("https://"):
            is_webcam = True
            self.logger.info(f"Kết nối tới nguồn luồng trực tuyến mạng: {source}...")
        else:
            self.logger.info(f"Đang phân tích file video nguồn: {source}...")
            if not os.path.exists(source):
                self.logger.error(f"Tệp video nguồn không tồn tại: {source}")
                sys.exit(1)
                
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.logger.error(f"Không thể truy cập dữ liệu video nguồn: {source}")
            sys.exit(1)
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames < 0:
            self.total_frames = 0
            
        self.logger.info(f"Độ phân giải: {width}x{height} | Tần số quét gốc: {fps:.2f} FPS | Tổng số frames: {self.total_frames}")
        
        # Khởi tạo bộ ghi Video đối với chế độ xử lý xuất file (Mode 1)
        if self.mode == 1:
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            self.h264_fallback = False
            if self.save_h264:
                try:
                    self.logger.info("Đang thử sử dụng codec H.264 (avc1) để ghi video kết quả...")
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                    if self.writer.isOpened():
                        self.logger.info(f"Đã khởi tạo ghi video H.264 thành công. File đích: {self.output_path}")
                    else:
                        self.logger.warning("Codec H.264 (avc1) không khả dụng. Tự động chuyển về codec mp4v...")
                        self.writer = None
                        self.h264_fallback = True
                except Exception as e:
                    self.logger.warning(f"Lỗi khởi tạo codec H.264: {e}. Đang chuyển về mp4v mặc định...")
                    self.writer = None
                    self.h264_fallback = True
            
            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                self.logger.info(f"Khởi tạo ghi video codec mp4v thành công. File đích: {self.output_path}")
                if self.save_h264:
                    self.h264_fallback = True
                
        # Khởi tạo cửa sổ xem hiển thị GUI
        if self.show_display:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self.logger.info("Đã mở cửa sổ hiển thị OpenCV GUI.")
            except Exception as e:
                self.logger.warning(f"Không thể hiển thị giao diện đồ họa: {e}. Tiến hành chạy ở chế độ ẩn (HEADLESS).")
                self.show_display = False
                
        self.logger.info("=== BẮT ĐẦU QUÁ TRÌNH PHÂN TÍCH NHẬN DIỆN ===")
        if self.show_display:
            print("Phím điều khiển: [Space/P] Tạm dừng/Tiếp tục | [S] Chụp màn hình thủ công | [Q/Esc] Thoát\n")
            
        paused = False
        
        while self.cap.isOpened():
            if paused:
                if self.show_display:
                    try:
                        key = cv2.waitKey(30) & 0xFF
                        if key in [ord(' '), ord('p'), ord('P')]:
                            paused = False
                            sys.stdout.write("\n")
                            self.logger.info("Tiến trình đã được người dùng tiếp tục.")
                        elif key in [ord('q'), ord('Q'), 27]:
                            self.logger.info("Tiến trình đã bị người dùng hủy bỏ.")
                            break
                    except Exception as e:
                        self.logger.warning(f"Sự cố giao diện khi đang tạm dừng: {e}. Tắt chế độ hiển thị GUI.")
                        self.show_display = False
                        paused = False
                else:
                    paused = False
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                break
                
            start_time = time.time()
            self.processed_frames += 1
            
            # Chạy suy luận nhận diện thông qua mô hình YOLO
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            inference_time = (time.time() - start_time) * 1000.0
            self.total_inference_time += inference_time
            
            detections = results[0]
            boxes = detections.boxes
            
            fire_count = 0
            smoke_count = 0
            annotated_frame = frame.copy()
            
            # Phân tích từng bounding box
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf_val = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                # 1. Áp dụng ngưỡng tin cậy riêng biệt cho từng lớp đối tượng
                required_conf = self.conf_threshold_fire if cls_id == 0 else self.conf_threshold_smoke
                if conf_val < required_conf:
                    continue
                    
                # 2. Trích xuất vùng ảnh nhận diện (ROI) để lọc báo động giả
                x1, y1, x2, y2 = xyxy
                h_f, w_f = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue
                    
                roi = frame[y1:y2, x1:x2]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                mean_brightness = gray_roi.mean()
                
                # BỘ LỌC PHỤ 1: Loại bỏ vùng quá tối hoàn toàn (ví dụ: khoảng không vũ trụ màu đen)
                # Lửa/khói thực tế cần ánh sáng hoặc phát sáng để camera thấy được.
                if mean_brightness < 18.0:
                    continue
                    
                # BỘ LỌC PHỤ 2: Lọc phát hiện Lửa giả trên bề mặt phẳng, đơn sắc (tường xám, nền trời xanh, v.v.)
                if cls_id == 0:  # Lửa
                    # Tính giá trị màu trung bình các kênh BGR
                    mean_b, mean_g, mean_r = cv2.mean(roi)[:3]
                    
                    # Lửa thật là nguồn sáng nóng, có sắc đỏ/cam/vàng (Red > Blue). 
                    # Nếu kênh Blue lớn hơn hoặc bằng kênh Red và độ tin cậy không cực kỳ cao (>85%), bỏ qua.
                    if mean_r <= mean_b and conf_val < 0.85:
                        continue
                        
                    # Lửa thật luôn có điểm phát sáng cực đại cao. Nếu độ sáng tối đa quá thấp, bỏ qua.
                    max_brightness = gray_roi.max()
                    if max_brightness < 150 and conf_val < 0.85:
                        continue
                
                class_label = self.model.names.get(cls_id, f"Class {cls_id}")
                # Nhãn hiển thị tiếng Việt không dấu trên màn hình OpenCV
                display_label = "Lua" if class_label == "fire" else ("Khoi" if class_label == "smoke" else class_label)
                label_text = f"{display_label} {conf_val:.2%}"
                
                if cls_id == 0:  # Lửa
                    fire_count += 1
                    color = (0, 0, 255)  # Màu đỏ
                elif cls_id == 1:  # Khói
                    smoke_count += 1
                    color = (0, 140, 255)  # Màu cam
                else:
                    color = (255, 255, 255)
                    
                self.draw_futuristic_box(
                    annotated_frame, 
                    xyxy, 
                    label_text, 
                    color, 
                    thickness=self.box_thickness, 
                    corner_len=self.corner_length
                )
                
            self.max_fire_count = max(self.max_fire_count, fire_count)
            self.max_smoke_count = max(self.max_smoke_count, smoke_count)
            
            # Logic tính toán chuỗi kích hoạt cảnh báo (Chỉ chạy khi enable_alerts = True)
            if self.enable_alerts:
                has_threat = (fire_count > 0 or smoke_count > 0)
                if has_threat:
                    self.consecutive_detections += 1
                else:
                    self.consecutive_detections = 0
                    
                if self.consecutive_detections >= self.alert_frames_threshold:
                    self.alert_triggered = True
                else:
                    self.alert_triggered = False
                    
                # Log sự kiện cảnh báo và thực hiện chụp ảnh lưu trữ (có Cooldown tránh ngập thư mục)
                if self.alert_triggered:
                    if not self.alert_event_active:
                        sys.stdout.write("\n")
                        self.logger.warning(f"🚨 [CẢNH BÁO KÍCH HOẠT] Tại Frame {self.processed_frames}: Phát hiện dấu hiệu hỏa hoạn!")
                        self.alert_event_active = True
                        
                    if self.save_alert_images and (self.processed_frames - self.last_alert_save_frame >= self.alert_save_cooldown_frames):
                        self.save_alert_screenshot(annotated_frame, fire_count, smoke_count)
                        self.last_alert_save_frame = self.processed_frames
                else:
                    if self.alert_event_active:
                        sys.stdout.write("\n")
                        self.logger.info(f"✅ [AN TOÀN TRỞ LẠI] Tại Frame {self.processed_frames}: Khu vực không còn lửa/khói nguy hiểm.")
                        self.alert_event_active = False
            else:
                self.alert_triggered = False
                
            # Tính toán hiệu năng thực tế
            end_time = time.time()
            frame_time = end_time - start_time
            self.total_fps_time += frame_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
            
            # Vẽ thanh điều khiển HUD
            self.draw_hud(annotated_frame, fire_count, smoke_count, inference_time, current_fps)
            
            # Ghi video đầu ra nếu chạy Chế độ 1 (Video File)
            if self.mode == 1 and self.writer is not None:
                self.writer.write(annotated_frame)
                
            # Tính toán độ trễ hiển thị nếu chạy mô phỏng Real-Time bằng file video ở Chế độ 2
            delay_ms = 1
            if self.mode == 2 and not is_webcam:
                elapsed_ms = (time.time() - start_time) * 1000.0
                target_delay = 1000.0 / fps
                delay_ms = int(max(1, target_delay - elapsed_ms))
                
            # Hiển thị trực quan cửa sổ đồ họa
            if self.show_display:
                try:
                    display_frame = annotated_frame
                    if width > 1280:
                        scale = 1280.0 / width
                        display_frame = cv2.resize(annotated_frame, (1280, int(height * scale)))
                        
                    cv2.imshow(self.window_name, display_frame)
                    
                    key = cv2.waitKey(delay_ms) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:
                        sys.stdout.write("\n")
                        self.logger.info("Tiến trình kết thúc theo yêu cầu người dùng.")
                        break
                    elif key in [ord(' '), ord('p'), ord('P')]:
                        paused = True
                        sys.stdout.write("\n")
                        self.logger.info("Tạm dừng xử lý. Ấn [Space] hoặc [P] để khôi phục chạy.")
                    elif key in [ord('s'), ord('S')]:
                        # Tạo thư mục theo ngày
                        date_folder = time.strftime("%Y-%m-%d")
                        target_dir = os.path.join("alerts", date_folder)
                        os.makedirs(target_dir, exist_ok=True)
                        
                        suffix = time.strftime("%d%m_%H%M%S")
                        raw_filename = os.path.join(target_dir, f"manual_raw_{self.processed_frames}_{suffix}.jpg")
                        cv2.imwrite(raw_filename, frame)
                        annotated_filename = os.path.join(target_dir, f"manual_annotated_{self.processed_frames}_{suffix}.jpg")
                        cv2.imwrite(annotated_filename, annotated_frame)
                        
                        abs_raw = os.path.abspath(raw_filename)
                        abs_annotated = os.path.abspath(annotated_filename)
                        sys.stdout.write("\n")
                        self.logger.info(f"📸 Đã lưu ảnh chụp thủ công:\n  - Gốc: {abs_raw}\n  - Vẽ hộp: {abs_annotated}")
                except Exception as e:
                    self.logger.warning(f"Cửa sổ đồ họa gặp sự cố: {e}. Tự động chuyển qua chạy ẩn (HEADLESS).")
                    self.show_display = False
                    if delay_ms > 1:
                        time.sleep(delay_ms / 1000.0)
            else:
                # Nếu chạy không màn hình ở Mode 2, thực hiện trễ mô phỏng để đồng bộ tốc độ phát video gốc
                if delay_ms > 1:
                    time.sleep(delay_ms / 1000.0)
                    
            # In tiến độ chạy theo hàng ngang trên terminal
            total_lbl = f"/{self.total_frames}" if self.total_frames > 0 else ""
            progress = (self.processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0.5
            prog_lbl = f" ({progress:.1f}%)" if self.total_frames > 0 else ""
            status_lbl = "🚨 CANH BAO" if self.alert_triggered else "AN TOAN"
            
            sys.stdout.write(
                f"\rKhung hinh: {self.processed_frames}{total_lbl}{prog_lbl} | "
                f"Lua: {fire_count} | Khoi: {smoke_count} | "
                f"Do tre: {inference_time:.1f}ms | Tốc độ: {current_fps:.1f} FPS | "
                f"Trang thai: {status_lbl}"
            )
            sys.stdout.flush()
            
        sys.stdout.write("\n")
        self.logger.info("Hoàn tất tiến trình chạy.")
        self.cleanup()
        self.print_summary()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
            
            # Tự động convert video sang chuẩn H.264 bằng ffmpeg nếu OpenCV phải fallback sang mp4v
            if getattr(self, 'h264_fallback', False) and os.path.exists(self.output_path):
                h264_path = self.output_path.replace(".mp4", "_h264.mp4")
                self.logger.info("Đang convert video sang H.264 bằng ffmpeg...")
                import subprocess
                try:
                    result = subprocess.run(
                        [
                            "ffmpeg", "-y",
                            "-i", self.output_path,
                            "-vcodec", "libx264",
                            "-crf", "23",
                            "-preset", "fast",
                            h264_path
                        ],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        self.logger.info(f"Đã convert H.264 thành công: {h264_path}")
                        self.h264_output_path = h264_path
                    else:
                        self.logger.warning("Convert H.264 bằng ffmpeg thất bại.")
                        if result.stderr:
                            self.logger.warning(f"Lỗi ffmpeg: {result.stderr[-300:].strip()}")
                except Exception as e:
                    self.logger.warning(f"Không thể chạy ffmpeg: {e}")
                    
        if self.show_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def print_summary(self):
        avg_inf = (self.total_inference_time / self.processed_frames) if self.processed_frames > 0 else 0.0
        avg_fps = (self.processed_frames / self.total_fps_time) if self.total_fps_time > 0 else 0.0
        
        summary_str = (
            f"\n" + "=" * 55 + "\n"
            f"          BÁO CÁO TỔNG KẾT HIỆU NĂNG NHẬN DIỆN          \n"
            f"=" * 55 + "\n"
            f"Số khung hình đã xử lý : {self.processed_frames}\n"
            f"Độ trễ xử lý trung bình: {avg_inf:.2f} ms\n"
            f"Tốc độ trung bình hệ thống: {avg_fps:.2f} FPS\n"
            f"Số lượng Lửa lớn nhất/khung hình : {self.max_fire_count}\n"
            f"Số lượng Khói lớn nhất/khung hình: {self.max_smoke_count}\n"
            f"Đánh giá rủi ro phiên chạy: {'🚨 PHÁT HIỆN NGUY HIỂM' if (self.max_fire_count > 0 or self.max_smoke_count > 0) else '✅ KHU VỰC AN TOÀN'}\n"
        )
        if self.log_file:
            summary_str += f"Tệp nhật ký (File Log) lưu tại  : {os.path.abspath(self.log_file)}\n"
        else:
            summary_str += f"Tệp nhật ký (File Log)          : ĐÃ TẮT GHI FILE LOG\n"
            
        if self.mode == 1 and self.writer is not None:
            summary_str += f"Video kết quả xuất ra lưu tại  : {os.path.abspath(self.output_path)}\n"
            if getattr(self, 'h264_output_path', None) and os.path.exists(self.h264_output_path):
                summary_str += f"Video H.264 (xem trên VS Code) : {os.path.abspath(self.h264_output_path)}\n"
        summary_str += "=" * 55 + "\n"
        
        # Ghi tóm tắt vào log file và in ra console
        for line in summary_str.split("\n"):
            if line.strip():
                self.logger.info(line)
        print(summary_str)

if __name__ == "__main__":
    engine = FireSmokeDetectionEngine(config_path="config.txt")
    engine.load_model()
    engine.run()