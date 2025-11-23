@echo off
echo Building 3D ASMR for Windows...

REM Install required packages
uv add nuitka zstandard

REM Build with Nuitka for Windows
python -m nuitka ^
  --standalone ^
  --enable-plugin=pyside6,numpy ^
  --include-data-dir=assets=assets ^
  --output-filename="3DASMR.exe" ^
  --assume-yes-for-downloads ^
  --output-dir=output ^
  --follow-import-to=need ^
  main.py

echo Build completed! Check the output directory.
pause