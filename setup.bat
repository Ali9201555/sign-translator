@echo off
setlocal

echo === Sign Translator Setup (no training needed) ===
echo.

where py >nul 2>nul
if %errorlevel%==0 (
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
    echo Failed to create venv. Install Python 3.10-3.12 from python.org and re-run.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo.

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Install failed. If mediapipe was the problem, install Python 3.12 from python.org.
    pause
    exit /b 1
)

echo.
echo === Done ===
echo Run it now with:    python translate.py
echo Or in a new shell:  .venv\Scripts\activate ^&^& python translate.py
echo.
pause
