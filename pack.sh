uv add nuitka zstandard
python -m nuitka --onefile --windows-disable-console --enable-plugin=pyside6,numpy --follow-import-to=need --output-dir=output --include-data-dir=assets=assets main.py


python -m nuitka \
  --standalone \
  --macos-create-app-bundle \
  --enable-plugin=pyside6,numpy \
  --macos-app-name="3D ASMR" \
  --company-name="xeelee.ai" \
  --macos-app-version="1.0.0" \
  --macos-signed-app-name="ai.xeelee.threedasmr" \
  --macos-sign-identity="Q6559HL64L" \
  --macos-sign-notarization \
  --assume-yes-for-downloads \
  --output-filename="3DASMR" \
  --macos-app-icon="AppIcons/appstore.icns" \
  --follow-import-to=need \
  --output-dir=output \
  --include-data-dir=assets=assets  \
  --macos-app-protected-resource="NSMicrophoneUsageDescription:Need microphone access for ASMR recording." \
  --script-name=main.py