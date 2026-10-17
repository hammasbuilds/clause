@echo off
REM Start the clause desk. Takes --port; opens no browser window of its own.
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
uv run --python 3.11 --no-project python -m clause.app %*
