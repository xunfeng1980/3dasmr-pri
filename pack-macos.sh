#!/bin/bash
echo "Building 3D ASMR for macOS..."

source ./venv/bin/activate

# Build with Nuitka for macOS as a single file
python -m nuitka \
  --onefile \
  --enable-plugin=pyside6,numpy \
  --output-filename="3DASMR" \
  --include-data-dir=assets=assets \
  --assume-yes-for-downloads \
  --output-dir=output \
  --follow-import-to=need \
  main.py

echo "Build completed! Check the output directory."

# Alternative: Build as an app bundle (uncomment to use this instead of single file)
# python -m nuitka \
#   --standalone \
#   --macos-create-app-bundle \
#   --enable-plugin=pyside6,numpy \
#   --macos-app-name="3D ASMR" \
#   --company-name="xeelee.ai" \
#   --macos-app-version="1.0.0" \
#   --macos-signed-app-name="ai.xeelee.threedasmr" \
#   --macos-sign-identity="Q6559HL64L" \
#   --macos-sign-notarization \
#   --assume-yes-for-downloads \
#   --output-filename="3DASMR" \
#   --macos-app-icon="AppIcons/appstore.icns" \
#   --follow-import-to=need \
#   --output-dir=output \
#   --include-data-dir=assets=assets  \
#   --macos-app-protected-resource="NSMicrophoneUsageDescription:Need microphone access for ASMR recording." \
#   --script-name=main.py