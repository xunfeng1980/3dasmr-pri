import sys
import numpy as np
import sounddevice as sd
import threading
import time
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QSlider, QPushButton,
                               QCheckBox, QSpinBox, QGroupBox)
from PySide6.QtCore import Signal, QObject, Qt
from PySide6.QtGui import QFont


class AudioProcessor(QObject):
    # Signal to update GUI from audio thread
    position_updated = Signal(float, float)

    def __init__(self):
        super().__init__()
        
        # Configuration
        self.SAMPLE_RATE = 48000
        self.BLOCK_SIZE = 512
        self.CHANNELS = 2
        self.HRTF_FILES = {
            "params": "assets/hrtf_params.txt",
            "left": "assets/hrtf_left.f32",
            "right": "assets/hrtf_right.f32"
        }

        # Check if files exist
        for name, path in self.HRTF_FILES.items():
            if not os.path.exists(path):
                print(f"❌ 找不到 HRTF 文件: {path}")
                sys.exit(1)

        # Load pre-computed HRTF table
        print("正在加载预计算 HRTF 表...")
        self.params = np.loadtxt(self.HRTF_FILES["params"], delimiter=",", dtype=float)

        file_size = os.path.getsize(self.HRTF_FILES["left"])
        self.n_measure = len(self.params)
        self.fir_length = file_size // (4 * self.n_measure)

        self.left_fir = np.fromfile(self.HRTF_FILES["left"], dtype=np.float32).reshape(self.n_measure, self.fir_length)
        self.right_fir = np.fromfile(self.HRTF_FILES["right"], dtype=np.float32).reshape(self.n_measure, self.fir_length)

        print(f"✅ 加载成功: {self.n_measure} 个方向, FIR 长度 = {self.fir_length}")

        # Global state
        self.azimuth = 0.0
        self.elevation = 0.0
        self.running = True
        self.auto_rotate = False
        self.rotate_speed = 30.0
        self.last_update_time = time.time()
        self.key_state = {"a": False, "d": False, "w": False, "s": False}

        self.overlap_buffer_left = np.zeros(self.fir_length - 1, dtype=np.float32)
        self.overlap_buffer_right = np.zeros(self.fir_length - 1, dtype=np.float32)

        # Find closest HRTF
        self.current_left_ir, self.current_right_ir = self.find_closest_hrtf(self.azimuth, self.elevation)
        
        # Find audio devices
        devices = sd.query_devices()
        self.bh_idx = next((i for i, d in enumerate(devices)
                           if "BlackHole 2ch" in d['name'] and d['max_input_channels'] >= 2), None)

        if self.bh_idx is None:
            print("❌ 未找到 BlackHole 2ch。请确认已安装并设为系统输出。")
            sys.exit(1)

        self.out_idx = next((i for i, d in enumerate(devices)
                            if "BlackHole" not in d['name'] and d['max_input_channels'] == 0 and d['max_output_channels'] >= 2), None)

        print(f"✅ 输入: {devices[self.bh_idx]['name']}")
        print(f"🔊 输出: {devices[self.out_idx]['name']}")

    def find_closest_hrtf(self, az, el):
        """返回最接近 (az, el) 的 (left_fir, right_fir)"""
        az_norm = az % 360
        diff_az = np.abs(self.params[:, 1] - az_norm)
        diff_az = np.minimum(diff_az, 360 - diff_az)
        diff_el = np.abs(self.params[:, 2] - el)
        distances = np.sqrt(diff_az**2 + diff_el**2)
        idx = np.argmin(distances)
        return self.left_fir[idx].copy(), self.right_fir[idx].copy()

    def process_audio(self, indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        if np.max(np.abs(indata)) < 1e-4:
            outdata.fill(0.0)
            return

        # Stereo remapping
        L_az = (self.azimuth - 30) % 360
        R_az = (self.azimuth + 30) % 360

        L_ir_L, L_ir_R = self.find_closest_hrtf(L_az, self.elevation)
        R_ir_L, R_ir_R = self.find_closest_hrtf(R_az, self.elevation)

        left_from_L = np.convolve(indata[:, 0], L_ir_L, mode='full')
        left_from_R = np.convolve(indata[:, 1], R_ir_L, mode='full')

        right_from_L = np.convolve(indata[:, 0], L_ir_R, mode='full')
        right_from_R = np.convolve(indata[:, 1], R_ir_R, mode='full')

        left_full = left_from_L + left_from_R
        right_full = right_from_L + right_from_R

        left_out = np.zeros(frames, dtype=np.float32)
        right_out = np.zeros(frames, dtype=np.float32)

        overlap_len = len(self.overlap_buffer_left)
        left_out[:overlap_len] = self.overlap_buffer_left
        right_out[:overlap_len] = self.overlap_buffer_right

        left_out += left_full[:frames]
        right_out += right_full[:frames]

        if len(left_full) > frames:
            self.overlap_buffer_left[:] = left_full[frames:frames+overlap_len]
            self.overlap_buffer_right[:] = right_full[frames:frames+overlap_len]
        else:
            self.overlap_buffer_left.fill(0)
            self.overlap_buffer_right.fill(0)

        gain = 0.3
        outdata[:, 0] = np.clip(left_out * gain, -1.0, 1.0)
        outdata[:, 1] = np.clip(right_out * gain, -1.0, 1.0)

    def start_audio_stream(self):
        self.stream = sd.Stream(
            device=(self.bh_idx, self.out_idx),
            channels=self.CHANNELS,
            samplerate=self.SAMPLE_RATE,
            blocksize=self.BLOCK_SIZE,
            dtype='float32',
            callback=self.process_audio
        )
        self.stream.start()

    def stop_audio_stream(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

    def update_position(self, new_azimuth, new_elevation):
        """外部接口：更新方位角和仰角"""
        self.azimuth = new_azimuth % 360
        self.elevation = max(min(new_elevation, 90.0), -40.0)
        self.current_left_ir, self.current_right_ir = self.find_closest_hrtf(self.azimuth, self.elevation)
        self.position_updated.emit(self.azimuth, self.elevation)

    def toggle_auto_rotation(self, enabled):
        """外部接口：切换自动旋转"""
        self.auto_rotate = enabled

    def set_rotate_speed(self, speed):
        """外部接口：设置旋转速度"""
        self.rotate_speed = speed

    def reset_position(self):
        """外部接口：重置位置"""
        self.update_position(0.0, 0.0)
        self.auto_rotate = False

    def auto_rotate_thread_func(self):
        """自动旋转线程函数"""
        while self.running:
            if self.auto_rotate:
                current_time = time.time()
                delta_time = current_time - self.last_update_time
                self.last_update_time = current_time

                # 更新方位角
                self.azimuth = (self.azimuth + self.rotate_speed * delta_time) % 360
                self.current_left_ir, self.current_right_ir = self.find_closest_hrtf(self.azimuth, self.elevation)
                self.position_updated.emit(self.azimuth, self.elevation)
            else:
                self.last_update_time = time.time()

            time.sleep(0.05)  # 20Hz 更新


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Audio HRTF Controller")
        self.setGeometry(100, 100, 600, 500)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("3D Audio HRTF Controller")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_layout.addWidget(title_label)

        # Create control group
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout(control_group)

        # Azimuth control
        azimuth_layout = QHBoxLayout()
        azimuth_label = QLabel("Azimuth (°):")
        self.azimuth_label_display = QLabel("0.0")
        self.azimuth_slider = QSlider(Qt.Orientation.Horizontal)
        self.azimuth_slider.setRange(0, 360)
        self.azimuth_slider.setValue(0)
        self.azimuth_slider.valueChanged.connect(self.azimuth_changed)
        
        azimuth_layout.addWidget(azimuth_label)
        azimuth_layout.addWidget(self.azimuth_slider)
        azimuth_layout.addWidget(self.azimuth_label_display)
        control_layout.addLayout(azimuth_layout)

        # Elevation control
        elevation_layout = QHBoxLayout()
        elevation_label = QLabel("Elevation (°):")
        self.elevation_label_display = QLabel("-0.0")
        self.elevation_slider = QSlider(Qt.Orientation.Horizontal)
        self.elevation_slider.setRange(-40, 90)
        self.elevation_slider.setValue(0)
        self.elevation_slider.valueChanged.connect(self.elevation_changed)
        
        elevation_layout.addWidget(elevation_label)
        elevation_layout.addWidget(self.elevation_slider)
        elevation_layout.addWidget(self.elevation_label_display)
        control_layout.addLayout(elevation_layout)

        # Auto-rotate controls
        rotation_layout = QHBoxLayout()
        self.auto_rotate_checkbox = QCheckBox("Auto Rotate")
        self.auto_rotate_checkbox.stateChanged.connect(self.toggle_auto_rotation)
        
        rotate_speed_label = QLabel("Speed (°/s):")
        self.rotate_speed_spinbox = QSpinBox()
        self.rotate_speed_spinbox.setRange(1, 180)
        self.rotate_speed_spinbox.setValue(30)
        self.rotate_speed_spinbox.valueChanged.connect(self.rotate_speed_changed)
        
        rotation_layout.addWidget(self.auto_rotate_checkbox)
        rotation_layout.addWidget(rotate_speed_label)
        rotation_layout.addWidget(self.rotate_speed_spinbox)
        control_layout.addLayout(rotation_layout)

        # Reset button
        self.reset_button = QPushButton("Reset Position")
        self.reset_button.clicked.connect(self.reset_position)
        control_layout.addWidget(self.reset_button)

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.status_label)

        main_layout.addWidget(control_group)

        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        
        # Connect position update signal
        self.audio_processor.position_updated.connect(self.update_position_display)
        
        # Start audio stream
        self.audio_processor.start_audio_stream()
        
        # Start auto-rotate thread
        self.auto_rotate_thread = threading.Thread(target=self.audio_processor.auto_rotate_thread_func, daemon=True)
        self.auto_rotate_thread.start()

        print("按界面按钮控制环绕声方向 | 使用滑块调整方位角和仰角")

    def azimuth_changed(self, value):
        self.azimuth_label_display.setText(f"{value:.1f}")
        self.audio_processor.update_position(value, self.audio_processor.elevation)

    def elevation_changed(self, value):
        self.elevation_label_display.setText(f"{value:.1f}")
        self.audio_processor.update_position(self.audio_processor.azimuth, value)

    def toggle_auto_rotation(self, state):
        self.audio_processor.toggle_auto_rotation(state == 2)  # Qt.Checked = 2
        status = "Enabled" if state == 2 else "Disabled"
        self.status_label.setText(f"Status: Auto Rotation {status}")

    def rotate_speed_changed(self, value):
        self.audio_processor.set_rotate_speed(float(value))

    def reset_position(self):
        self.audio_processor.reset_position()
        self.azimuth_slider.setValue(0)
        self.elevation_slider.setValue(0)
        self.auto_rotate_checkbox.setChecked(False)
        self.status_label.setText("Status: Position Reset")

    def update_position_display(self, azimuth, elevation):
        # Update sliders and display labels from audio thread
        self.azimuth_slider.setValue(int(azimuth))
        self.elevation_slider.setValue(int(elevation))
        self.azimuth_label_display.setText(f"{azimuth:.1f}")
        self.elevation_label_display.setText(f"{elevation:.1f}")

    def closeEvent(self, event):
        # Stop audio stream on close
        self.audio_processor.running = False
        self.audio_processor.stop_audio_stream()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()