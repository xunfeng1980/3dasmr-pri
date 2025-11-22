uv add nuitka zstandard
python -m nuitka --onefile --windows-disable-console --enable-plugin=pyside6,numpy --follow-import-to=need --output-dir=output --include-data-dir=assets=assets main.py