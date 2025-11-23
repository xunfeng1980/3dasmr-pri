import sys
import numpy as np
import sounddevice as sd
import threading
import time
import os
import platform
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QSlider, QPushButton,
                               QCheckBox, QSpinBox, QGroupBox, QComboBox)
from PySide6.QtCore import Signal, QObject, Qt
from PySide6.QtGui import QFont


class AudioProcessor(QObject):
    position_updated = Signal(float, float)

    def __init__(self, out_idx=None):
        super().__init__()

        self.SAMPLE_RATE = 48000
        self.BLOCK_SIZE = 512
        self.CHANNELS = 2
        self.HRTF_FILES = {
            "params": "assets/hrtf_params.txt",
            "left": "assets/hrtf_left.f32",
            "right": "assets/hrtf_right.f32"
        }

        for name, path in self.HRTF_FILES.items():
            if not os.path.exists(path):
                print(f"❌ HRTF file not found: {path}")
                sys.exit(1)

        print("Loading precomputed HRTF tables...")
        self.params = np.loadtxt(self.HRTF_FILES["params"], delimiter=",", dtype=float)

        file_size = os.path.getsize(self.HRTF_FILES["left"])
        self.n_measure = len(self.params)
        self.fir_length = file_size // (4 * self.n_measure)

        self.left_fir = np.fromfile(self.HRTF_FILES["left"], dtype=np.float32).reshape(self.n_measure, self.fir_length)
        self.right_fir = np.fromfile(self.HRTF_FILES["right"], dtype=np.float32).reshape(self.n_measure, self.fir_length)

        print(f"✅ Loaded: {self.n_measure} directions, FIR length = {self.fir_length}")

        self.azimuth = 0.0
        self.elevation = 0.0
        self.running = True
        self.auto_rotate = True  # ←←← 默认开启自动旋转
        self.rotate_speed = 30.0
        self.last_update_time = time.time()

        self.overlap_buffer_left = np.zeros(self.fir_length - 1, dtype=np.float32)
        self.overlap_buffer_right = np.zeros(self.fir_length - 1, dtype=np.float32)

        self.current_left_ir, self.current_right_ir = self.find_closest_hrtf(self.azimuth, self.elevation)

        # --- Platform-specific device setup ---
        devices = sd.query_devices()
        current_platform = platform.system()

        if current_platform == "Windows":
            hostapis = sd.query_hostapis()
            wasapi_id = None
            for i, api in enumerate(hostapis):
                if api['name'] == 'Windows WASAPI':
                    wasapi_id = i
                    break
            if wasapi_id is None:
                print("❌ WASAPI host API not found.")
                sys.exit(1)

            default_output = sd.default.device[1]
            if default_output is None:
                output_devices = [i for i, d in enumerate(devices) if d['max_output_channels'] >= 2]
                default_output = output_devices[0] if output_devices else None

            if default_output is None:
                print("❌ No output device available for WASAPI loopback.")
                sys.exit(1)

            self.loopback_device = default_output
            self.wasapi_id = wasapi_id
            self.input_mode = "WASAPI Loopback"

        elif current_platform == "Darwin":
            self.loopback_device = next((i for i, d in enumerate(devices)
                                         if "BlackHole 2ch" in d['name'] and d['max_input_channels'] >= 2), None)
            if self.loopback_device is None:
                print("❌ BlackHole 2ch not found. Please install BlackHole (https://existential.audio/blackhole/).")
                sys.exit(1)
            self.wasapi_id = None
            self.input_mode = "BlackHole"
        else:
            print("❌ Only Windows and macOS are supported.")
            sys.exit(1)

        # Output device
        if out_idx is not None:
            self.out_idx = out_idx
        else:
            candidates = [i for i, d in enumerate(devices)
                          if d['max_output_channels'] >= 2 and d['max_input_channels'] == 0]
            self.out_idx = candidates[0] if candidates else None

        if self.out_idx is None:
            print("❌ No valid output audio device found.")
            sys.exit(1)

        dev_name = devices[self.out_idx]['name']
        print(f"🔊 Output device: {dev_name}")

    def find_closest_hrtf(self, az, el):
        az_norm = az % 360
        diff_az = np.abs(self.params[:, 1] - az_norm)
        diff_az = np.minimum(diff_az, 360 - diff_az)
        diff_el = np.abs(self.params[:, 2] - el)
        distances = np.sqrt(diff_az ** 2 + diff_el ** 2)
        idx = np.argmin(distances)
        return self.left_fir[idx].copy(), self.right_fir[idx].copy()

    def process_audio_block(self, indata):
        frames = indata.shape[0]
        if np.max(np.abs(indata)) < 1e-4:
            return np.zeros((frames, 2), dtype=np.float32)

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
            self.overlap_buffer_left[:] = left_full[frames:frames + overlap_len]
            self.overlap_buffer_right[:] = right_full[frames:frames + overlap_len]
        else:
            self.overlap_buffer_left.fill(0)
            self.overlap_buffer_right.fill(0)

        gain = 0.3
        outdata = np.column_stack((
            np.clip(left_out * gain, -1.0, 1.0),
            np.clip(right_out * gain, -1.0, 1.0)
        ))
        return outdata

    def input_callback(self, indata, frames, time_info, status):
        if status:
            print("Input stream status:", status, file=sys.stderr)
        outdata = self.process_audio_block(indata)
        if hasattr(self, 'output_stream') and self.output_stream.active:
            try:
                self.output_stream.write(outdata)
            except Exception as e:
                print(f"⚠️ Output write error: {e}", file=sys.stderr)

    def start_audio_stream(self):
        current_platform = platform.system()

        self.output_stream = sd.OutputStream(
            device=self.out_idx,
            channels=self.CHANNELS,
            samplerate=self.SAMPLE_RATE,
            blocksize=self.BLOCK_SIZE,
            dtype='float32'
        )
        self.output_stream.start()

        if current_platform == "Windows":
            self.input_stream = sd.InputStream(
                device=self.loopback_device,
                host_api=self.wasapi_id,
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                dtype='float32',
                callback=self.input_callback,
                latency='low',
                extra_settings=sd.WasapiSettings(loopback=True)
            )
        elif current_platform == "Darwin":
            self.input_stream = sd.InputStream(
                device=self.loopback_device,
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                dtype='float32',
                callback=self.input_callback,
                latency='low'
            )
        else:
            raise RuntimeError("Unsupported platform")

        self.input_stream.start()

    def stop_audio_stream(self):
        for name in ['input_stream', 'output_stream']:
            if hasattr(self, name):
                stream = getattr(self, name)
                try:
                    if stream.active:
                        stream.stop()
                    stream.close()
                except:
                    pass

    def change_output_device(self, new_out_idx):
        devices = sd.query_devices()
        if not (0 <= new_out_idx < len(devices)):
            return
        d = devices[new_out_idx]
        if d['max_output_channels'] < 2 or d['max_input_channels'] != 0:
            return
        if new_out_idx != self.out_idx:
            self.out_idx = new_out_idx
            self.stop_audio_stream()
            self.start_audio_stream()

    def update_position(self, new_azimuth, new_elevation):
        self.azimuth = new_azimuth % 360
        self.elevation = max(min(new_elevation, 90.0), -40.0)
        self.position_updated.emit(self.azimuth, self.elevation)

    def toggle_auto_rotation(self, enabled):
        self.auto_rotate = enabled

    def set_rotate_speed(self, speed):
        self.rotate_speed = speed

    def reset_position(self):
        self.update_position(0.0, 0.0)
        self.auto_rotate = False

    def auto_rotate_thread_func(self):
        while self.running:
            if self.auto_rotate:
                dt = time.time() - self.last_update_time
                self.last_update_time = time.time()
                self.azimuth = (self.azimuth + self.rotate_speed * dt) % 360
                self.position_updated.emit(self.azimuth, self.elevation)
            else:
                self.last_update_time = time.time()
            time.sleep(0.05)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        current_platform = platform.system()
        if current_platform not in ("Windows", "Darwin"):
            print("Only Windows and macOS are supported.")
            sys.exit(1)

        self.setWindowTitle("3D Audio HRTF Controller")
        self.setGeometry(100, 100, 620, 540)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("3D Audio HRTF Controller")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        main_layout.addWidget(title_label)

        # Subtitle
        if current_platform == "Windows":
            subtitle_text = "Using WASAPI Loopback to capture system audio"
        else:  # macOS
            subtitle_text = "Using BlackHole to capture system audio"
        subtitle_label = QLabel(subtitle_text)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: gray; font-size: 12px;")
        main_layout.addWidget(subtitle_label)

        # Control group
        control_group = QGroupBox("Head Orientation")
        control_layout = QVBoxLayout(control_group)

        # Azimuth
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

        # Elevation
        elevation_layout = QHBoxLayout()
        elevation_label = QLabel("Elevation (°):")
        self.elevation_label_display = QLabel("0.0")
        self.elevation_slider = QSlider(Qt.Orientation.Horizontal)
        self.elevation_slider.setRange(-40, 90)
        self.elevation_slider.setValue(0)
        self.elevation_slider.valueChanged.connect(self.elevation_changed)
        elevation_layout.addWidget(elevation_label)
        elevation_layout.addWidget(self.elevation_slider)
        elevation_layout.addWidget(self.elevation_label_display)
        control_layout.addLayout(elevation_layout)

        # Auto rotation — DEFAULT ENABLED
        rotation_layout = QHBoxLayout()
        self.auto_rotate_checkbox = QCheckBox("Auto Rotate")
        self.auto_rotate_checkbox.setChecked(True)  # ←←← 默认勾选
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

        # Reset
        self.reset_button = QPushButton("Reset Position")
        self.reset_button.clicked.connect(self.reset_position)
        control_layout.addWidget(self.reset_button)

        # Output device
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Device:")
        self.output_combo = QComboBox()
        devices = sd.query_devices()
        output_devs = [
            (i, d['name']) for i, d in enumerate(devices)
            if d['max_output_channels'] >= 2 and d['max_input_channels'] == 0
        ]
        for i, name in output_devs:
            self.output_combo.addItem(name, i)

        default_out = sd.default.device[1]
        if default_out is not None:
            for idx in range(self.output_combo.count()):
                if self.output_combo.itemData(idx) == default_out:
                    self.output_combo.setCurrentIndex(idx)
                    break
        self.output_combo.currentIndexChanged.connect(self.on_output_device_changed)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_combo)
        control_layout.addLayout(output_layout)

        # Status label
        if current_platform == "Windows":
            status_text = "Status: Ready (Windows: Using WASAPI Loopback)"
        else:
            status_text = "Status: Ready (macOS: Using BlackHole)"
        self.status_label = QLabel(status_text)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.status_label)

        main_layout.addWidget(control_group)

        # Initialize processor
        initial_out = self.output_combo.itemData(self.output_combo.currentIndex()) if self.output_combo.count() > 0 else None
        self.audio_processor = AudioProcessor(out_idx=initial_out)
        self.audio_processor.position_updated.connect(self.update_gui_from_processor)
        self.audio_processor.start_audio_stream()

        self.rotate_thread = threading.Thread(target=self.audio_processor.auto_rotate_thread_func, daemon=True)
        self.rotate_thread.start()

        # Set initial status for auto rotation
        self.status_label.setText("Status: Auto Rotation ON")

    def azimuth_changed(self, value):
        self.azimuth_label_display.setText(f"{value:.1f}")
        self.audio_processor.update_position(value, self.audio_processor.elevation)

    def elevation_changed(self, value):
        self.elevation_label_display.setText(f"{value:.1f}")
        self.audio_processor.update_position(self.audio_processor.azimuth, value)

    def toggle_auto_rotation(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.audio_processor.toggle_auto_rotation(enabled)
        status = "ON" if enabled else "OFF"
        self.status_label.setText(f"Status: Auto Rotation {status}")

    def rotate_speed_changed(self, value):
        self.audio_processor.set_rotate_speed(float(value))

    def reset_position(self):
        self.audio_processor.reset_position()
        self.azimuth_slider.setValue(0)
        self.elevation_slider.setValue(0)
        self.auto_rotate_checkbox.setChecked(False)
        self.status_label.setText("Status: Position Reset")

    def update_gui_from_processor(self, azimuth, elevation):
        self.azimuth_slider.blockSignals(True)
        self.elevation_slider.blockSignals(True)
        self.azimuth_slider.setValue(int(azimuth))
        self.elevation_slider.setValue(int(elevation))
        self.azimuth_label_display.setText(f"{azimuth:.1f}")
        self.elevation_label_display.setText(f"{elevation:.1f}")
        self.azimuth_slider.blockSignals(False)
        self.elevation_slider.blockSignals(False)

    def on_output_device_changed(self, index):
        if index >= 0:
            dev_idx = self.output_combo.itemData(index)
            self.audio_processor.change_output_device(dev_idx)

    def closeEvent(self, event):
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