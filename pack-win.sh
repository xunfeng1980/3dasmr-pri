#!/bin/bash
echo "Building 3D ASMR for Windows..."

source ./venv/Scripts/activate

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

