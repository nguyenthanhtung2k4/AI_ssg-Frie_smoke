import cv2
import threading
import time
import os
from datetime import datetime

import urllib.parse

# ==============================================================================
#  CẤU HÌNH KẾT NỐI CAMERA HIKVISION
# ==============================================================================
IP_ADDRESS  = "192.168.1.165"     # Thay đổi thành IP của camera Hikvision
PORT        = 554                # Cổng RTSP mặc định là 554
USERNAME    = "admin"            # Tài khoản camera
PASSWORD    = "Nd@13579"     # Mật khẩu camera (ví dụ lấy từ config cũ của bạn)
CHANNEL     = 1                  # Số kênh (thường là 1)
STREAM_TYPE = 1                  # 1: Main Stream (Độ phân giải cao) | 2: Sub Stream (Nhẹ, mượt)

# Tự động tạo URL RTSP theo định dạng chuẩn của Hikvision (URL-encode tài khoản/mật khẩu để tránh lỗi ký tự đặc biệt như @, :, /):
# Ví dụ: rtsp://admin:Nd%4013579@192.168.1.165:554/Streaming/Channels/101
safe_username = urllib.parse.quote(USERNAME)
safe_password = urllib.parse.quote(PASSWORD)
RTSP_URL = f"rtsp://{safe_username}:{safe_password}@{IP_ADDRESS}:{PORT}/Streaming/Channels/{CHANNEL}0{STREAM_TYPE}"

# Thư mục lưu ảnh chụp màn hình từ camera
SCREENSHOT_DIR = "./screenshots"

# ==============================================================================
#  LỚP HỖ TRỢ ĐỌC RTSP ĐA LUỒNG (TRÁNH TRỄ / GIẬT LAG HÌNH ẢNH)
# ==============================================================================
class HikvisionStreamReader:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.last_update = time.time()
        self.fps = 0.0
        self.frame_count = 0
        self.is_connected = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _update_loop(self):
        last_fps_time = time.time()
        
        while self.running:
            # Nếu chưa kết nối hoặc kết nối bị mất, tiến hành kết nối lại
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                print(f"\n[CONNECTING] Dang ket noi toi: rtsp://{USERNAME}:******@{IP_ADDRESS}:{PORT}/...")
                
                # Khởi tạo VideoCapture với RTSP URL
                self.cap = cv2.VideoCapture(self.rtsp_url)
                
                if not self.cap.isOpened():
                    print("[FAILED] Ket noi camera that bai! Se thu lai sau 5 giay...")
                    time.sleep(5)
                    continue
                else:
                    self.is_connected = True
                    print("[SUCCESS] Ket noi thanh cong voi Camera Hikvision!")
                    self.frame_count = 0
                    last_fps_time = time.time()

            # Đọc frame tiếp theo từ buffer
            ret, frame = self.cap.read()
            
            if not ret:
                print("\n[WARNING] Khong the doc frame tu camera. Dang thu ket noi lai...")
                self.cap.release()
                self.cap = None
                self.is_connected = False
                time.sleep(1)
                continue

            # Cập nhật frame mới nhất một cách an toàn
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.last_update = time.time()
                self.frame_count += 1

            # Tính toán FPS thực tế nhận được
            now = time.time()
            elapsed = now - last_fps_time
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                last_fps_time = now

            # Tránh overload CPU (Chờ 1 khoảng cực ngắn phù hợp với camera 25-30fps)
            time.sleep(0.005)

    def read(self):
        with self.lock:
            if self.ret and self.frame is not None:
                # Trả về bản sao để tránh conflict thread vẽ lên frame
                return self.ret, self.frame.copy()
            return False, None

    def get_stats(self):
        with self.lock:
            return self.is_connected, self.fps

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
        print("[INFO] Da giai phong camera.")

# ==============================================================================
#  CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
def main():
    print("=" * 60)
    print("   HIKVISION CAMERA RTSP VIEWER & TESTER")
    print("=" * 60)
    print(f"IP Camera: {IP_ADDRESS}")
    print(f"Cong RTSP: {PORT}")
    print(f"Kenh: {CHANNEL} | Stream: {'Main Stream (HD)' if STREAM_TYPE == 1 else 'Sub Stream (SD)'}")
    print(f"Duong dan ket noi: {RTSP_URL}")
    print("-" * 60)
    print("Phim tat dieu khien:")
    print("  - ESC hoac 'q': Thoat chuong trinh")
    print("  - 's': Chup anh man hinh luu lai")
    print("=" * 60)

    # Tạo thư mục chụp ảnh màn hình nếu chưa có
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # Khởi động luồng đọc camera
    reader = HikvisionStreamReader(RTSP_URL).start()

    cv2.namedWindow("Hikvision Live Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hikvision Live Feed", 960, 540) # Kích thước cửa sổ preview phù hợp

    try:
        while True:
            # Lấy frame mới nhất
            ret, frame = reader.read()
            is_connected, live_fps = reader.get_stats()

            # Tạo màn hình chờ nếu chưa kết nối được
            if not ret or not is_connected:
                # Vẽ màn hình đen thông báo đang kết nối
                import numpy as np
                waiting_frame = np.zeros((540, 960, 3), dtype=np.uint8)
                
                cv2.putText(waiting_frame, "Dang ket noi camera Hikvision...", (50, 270), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(waiting_frame, f"RTSP URL: rtsp://{USERNAME}:***@{IP_ADDRESS}:{PORT}...", (50, 310), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
                cv2.imshow("Hikvision Live Feed", waiting_frame)
            else:
                h, w = frame.shape[:2]
                
                # Vẽ thanh trạng thái / thông số lên frame
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (380, 110), (0, 0, 0), -1)
                # Trộn overlay làm mờ nền
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Viết thông tin trạng thái lên màn hình
                cv2.putText(frame, f"STATUS: CONNECTED", (20, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {live_fps:.2f} frames/sec", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Resolution: {w}x{h}", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "[s] Chup anh | [q] Thoat", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

                # Hiển thị frame
                cv2.imshow("Hikvision Live Feed", frame)

            # Xử lý sự kiện nhấn phím
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC hoặc q
                print("\n[INFO] Dang dong ung dung...")
                break
            elif key == ord('s') and ret: # Nhấn 's' để chụp ảnh
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(SCREENSHOT_DIR, filename)
                cv2.imwrite(filepath, frame)
                print(f"\n[SAVED] Da luu anh chup man hinh tai: {filepath}")

            # Sleep nhỏ để giảm tải CPU cho main loop
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Da huy boi nguoi dung.")
    finally:
        # Dừng luồng đọc và đóng các cửa sổ
        reader.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
