@echo off
setlocal

echo === Sign Translator Setup ===
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    echo Found Python launcher. Trying py -3.12 first, then py -3.11, then default python.
    py -3.12 --version >nul 2>nul && (set PY=py -3.12) || (
        py -3.11 --version >nul 2>nul && (set PY=py -3.11) || (set PY=python)
    )
) else (
    set PY=python
)

echo Using interpreter: %PY%
%PY% --version
echo.

echo Creating virtual environment...
%PY% -m venv .venv
if errorlevel 1 (
    echo Failed to create venv. Make sure Python 3.10-3.12 is installed.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo.

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies (this can take a few minutes)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Install failed. If mediapipe was the problem, try installing Python 3.12 from python.org and re-run this script.
    pause
    exit /b 1
)

echo.
echo === Done ===
echo To use the venv in any new terminal:  .venv\Scripts\activate
echo.
echo Quick start:
echo   1. Record sign samples:    python collect_data.py hello
echo   2. Record a second sign:   python collect_data.py thanks
echo   3. Train the model:        python train.py
echo   4. Translate live:         python translate.py
echo.
pause
