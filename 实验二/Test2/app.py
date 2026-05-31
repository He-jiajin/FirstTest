import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage, Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, \
    QProgressBar, QSlider, QComboBox, QMainWindow, QSplitter
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO
from collections import defaultdict
import os

MODEL_PATH = './Model/yolov8n.pt'
YOLO_SEG_MODEL = './Model/yolov8n-seg.pt'

def cv2_to_qimage(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    qimage = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return qimage.rgbSwapped()

class VideoWorker(QThread):
    frame_processed = Signal(QImage, dict)
    progress_updated = Signal(int)
    finished = Signal()

    def __init__(self, video_path, model_path, function_type):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.function_type = function_type
        self.running = False
        self.paused = False
        self.frame_count = 0
        self.total_frames = 0
        self.fps = 30
        self.current_frame = None
        
        self.track_history = defaultdict(list)
        self.car_count = 0
        self.car_ids = set()

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        if self.function_type in ['car_tracking', 'traffic', 'speed']:
            model = YOLO(self.model_path)
        elif self.function_type == 'heatmap':
            model = YOLO(YOLO_SEG_MODEL)
        
        while self.running and cap.isOpened():
            while self.paused and self.running:
                self.msleep(100)
                continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            
            if self.function_type == 'car_tracking':
                annotated_frame = self.draw_tracking(results, frame)
            elif self.function_type == 'traffic':
                annotated_frame = self.draw_traffic(results, frame)
            elif self.function_type == 'speed':
                annotated_frame = self.draw_speed(results, frame)
            elif self.function_type == 'heatmap':
                annotated_frame = self.draw_heatmap(results, frame)
            
            qimage = cv2_to_qimage(annotated_frame)
            self.current_frame = annotated_frame
            
            stats = {
                'car_count': len(self.car_ids),
                'frame': self.frame_count,
                'total_frames': self.total_frames,
                'fps': self.fps
            }
            
            self.frame_processed.emit(qimage, stats)
            self.progress_updated.emit(int((self.frame_count / self.total_frames) * 100))
            self.frame_count += 1
            
            self.msleep(int(1000 / self.fps))
        
        cap.release()
        self.finished.emit()

    def draw_tracking(self, results, frame):
        annotated_frame = results[0].plot()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
                
                self.car_ids.add(track_id)
        
        return annotated_frame

    def draw_traffic(self, results, frame):
        annotated_frame = frame.copy()
        
        if results[0].boxes.id is not None:
            masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            self.car_ids = set(track_ids)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                track_id = track_ids[i]
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'ID: {track_id}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if masks is not None:
                    mask = masks[i]
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    color = (0, 0, 255)
                    annotated_frame[mask > 0.5] = annotated_frame[mask > 0.5] * 0.5 + np.array(color) * 0.5
        
        cv2.putText(annotated_frame, f'车辆数量: {len(self.car_ids)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return annotated_frame

    def draw_speed(self, results, frame):
        annotated_frame = results[0].plot()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                self.track_history[track_id].append((float(x), float(y)))
                self.car_ids.add(track_id)
                
                if len(self.track_history[track_id]) >= 2:
                    prev_x, prev_y = self.track_history[track_id][-2]
                    curr_x, curr_y = self.track_history[track_id][-1]
                    distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    speed = distance * self.fps * 0.1
                    
                    cv2.putText(annotated_frame, f'{speed:.1f} km/h', (int(x - w/2), int(y - h/2 - 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return annotated_frame

    def draw_heatmap(self, results, frame):
        heatmap = np.zeros_like(frame, dtype=np.float32)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                heatmap[y1:y2, x1:x2] += 1
        
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        annotated_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        
        cv2.putText(annotated_frame, '交通热图', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated_frame

    def stop(self):
        self.running = False

    def toggle_pause(self):
        self.paused = not self.paused

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('智能交通系统')
        self.setGeometry(100, 100, 1200, 800)
        
        self.video_path = None
        self.current_function = 'car_tracking'
        self.worker = None
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet('background-color: black;')
        left_layout.addWidget(self.video_label)
        
        control_layout = QHBoxLayout()
        
        self.select_btn = QPushButton('选择文件')
        self.select_btn.clicked.connect(self.select_video)
        control_layout.addWidget(self.select_btn)
        
        self.play_btn = QPushButton('播放')
        self.play_btn.clicked.connect(self.play_video)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton('暂停')
        self.pause_btn.clicked.connect(self.pause_video)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton('停止')
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(control_layout)
        
        function_layout = QHBoxLayout()
        function_layout.addWidget(QLabel('功能选择:'))
        
        self.function_combo = QComboBox()
        self.function_combo.addItems(['车辆追踪', '车流量检测', '速度检测', '道路热图'])
        self.function_combo.currentIndexChanged.connect(self.change_function)
        function_layout.addWidget(self.function_combo)
        
        left_layout.addLayout(function_layout)
        
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel('进度:'))
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        left_layout.addLayout(progress_layout)
        
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel('车辆数量: 0 | 帧: 0/0')
        stats_layout.addWidget(self.stats_label)
        
        left_layout.addLayout(stats_layout)
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel('帧率:'))
        
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 60)
        self.fps_slider.setValue(30)
        fps_layout.addWidget(self.fps_slider)
        
        self.fps_label = QLabel('30 FPS')
        fps_layout.addWidget(self.fps_label)
        
        self.fps_slider.valueChanged.connect(self.update_fps)
        
        left_layout.addLayout(fps_layout)
        
        main_layout.addLayout(left_layout)
        
        self.show()

    def select_video(self):
        file_dialog = QFileDialog()
        self.video_path, _ = file_dialog.getOpenFileName(self, '选择视频文件', '', '视频文件 (*.mp4 *.avi *.mov)')
        
        if self.video_path:
            self.play_btn.setEnabled(True)
            self.video_label.setText(f'已选择: {os.path.basename(self.video_path)}')

    def get_function_type(self):
        index = self.function_combo.currentIndex()
        if index == 0:
            return 'car_tracking'
        elif index == 1:
            return 'traffic'
        elif index == 2:
            return 'speed'
        elif index == 3:
            return 'heatmap'

    def play_video(self):
        if not self.video_path:
            return
        
        self.stop_video()
        
        self.worker = VideoWorker(self.video_path, MODEL_PATH, self.get_function_type())
        self.worker.frame_processed.connect(self.update_frame)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.select_btn.setEnabled(False)

    def pause_video(self):
        if self.worker:
            self.worker.toggle_pause()
            if self.worker.paused:
                self.pause_btn.setText('继续')
            else:
                self.pause_btn.setText('暂停')

    def stop_video(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.select_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.pause_btn.setText('暂停')

    def update_frame(self, qimage, stats):
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)
        
        self.stats_label.setText(f'车辆数量: {stats["car_count"]} | 帧: {stats["frame"]}/{stats["total_frames"]}')

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def change_function(self):
        if self.worker and self.worker.running:
            self.stop_video()

    def update_fps(self, value):
        self.fps_label.setText(f'{value} FPS')
        if self.worker:
            self.worker.fps = value

    def on_finished(self):
        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.select_btn.setEnabled(True)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec()
