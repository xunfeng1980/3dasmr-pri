#!/bin/bash
echo "Building 3D ASMR for Windows..."

python -m nuitka \
  --standalone \
  --enable-plugin=pyside6,numpy \
  --include-data-dir=assets=assets \
  --windows-disable-console \
  --output-filename="3DASMR" \
  --assume-yes-for-downloads \
  --output-dir=output \
  --follow-import-to=need \
  main.py
echo "Build completed! Check the output directory."

