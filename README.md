# 3D Audio HRTF Controller

A cross-platform 3D audio application that applies Head-Related Transfer Function (HRTF) processing to create spatial audio effects.

## Features
- Real-time 3D audio processing with HRTF
- Adjustable azimuth and elevation controls
- Auto-rotation functionality
- Cross-platform support (Windows, macOS, Linux)

## Installation

### Prerequisites
- Python 3.10 or higher
- [uvi](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   uv sync  # or use pip install -r requirements.txt if using pip
   ```

## Platform-Specific Setup

### Windows
On Windows, you have two options for audio routing:

#### Option 1: Virtual Audio Cable (Recommended)
Install one of these virtual audio solutions:
- [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free)
- [VoiceMeeter](https://vb-audio.com/Voicemeeter/) (free)

After installation, route your audio applications through the virtual cable for processing.

#### Option 2: WASAPI Loopback
The application can use WASAPI to capture system audio directly, but this method may not work with all applications and has some limitations.

### macOS
The application uses BlackHole 2ch virtual audio device by default. Install [BlackHole](https://existential.audio/blackhole/) for best results.

### Linux
The application supports JACK, PulseAudio, and standard ALSA audio routing. For virtual audio routing, consider using JACK.

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Configure your audio applications to output to the virtual audio device

3. Use the GUI to control the 3D audio positioning:
   - Adjust azimuth (horizontal rotation) slider
   - Adjust elevation (vertical angle) slider
   - Enable auto-rotation with adjustable speed
   - Select different output devices from the dropdown

## Building Executables

### Windows
```bash
# Using the batch script
pack.bat

# Or manually
python -m nuitka --standalone --enable-plugin=pyside6,numpy --windows-disable-console --include-data-dir=assets=assets --output-filename="3DASMR.exe" main.py
```

### macOS
```bash
# Using the shell script
chmod +x pack.sh
./pack.sh

# Or manually
python -m nuitka --standalone --macos-create-app-bundle --enable-plugin=pyside6,numpy --macos-app-name="3D ASMR" --include-data-dir=assets=assets --output-filename="3DASMR" main.py
```

## License
MIT License