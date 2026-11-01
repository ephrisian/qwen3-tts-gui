@echo off
setlocal enabledelayedexpansion

REM ========================================================================
REM install.bat - Windows installer for Qwen3-TTS GUI
REM
REM Requires Conda (Miniconda or Anaconda) to be installed and available in PATH.
REM Creates a Conda environment named "qwen3-tts" (Python 3.10) and installs
REM all project dependencies.
REM ========================================================================

set ENV_NAME=qwen3-tts
set SCRIPT_NAME=launch.py

echo Qwen3-TTS GUI - Windows Installer
echo ===================================

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: conda is not installed or not in PATH.
    echo Please install Miniconda or Anaconda first:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

REM Resolve the directory this script lives in
set SCRIPT_DIR=%~dp0
set SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_NAME%

REM Check if the environment already exists
conda env list | findstr /b /c:"%ENV_NAME%" >nul 2>&1
if %errorlevel% equ 0 (
    echo Conda environment '%ENV_NAME%' already exists.
) else (
    echo Creating Conda environment: %ENV_NAME% ...
    conda create -n %ENV_NAME% python=3.10 -y
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create conda environment.
        exit /b 1
    )
)

echo Activating environment ...
call conda activate %ENV_NAME%
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate conda environment '%ENV_NAME%'.
    exit /b 1
)

REM Install torch stack (cu121)
echo Installing torch/torchvision/torchaudio (cu121) ...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ERROR: Failed to install torch stack.
    exit /b 1
)

REM Install qwen-tts
echo Installing qwen-tts ...
pip install -U qwen-tts
if %errorlevel% neq 0 (
    echo ERROR: Failed to install qwen-tts.
    exit /b 1
)

REM Install flash-attn
echo Installing flash-attn ...
pip install -U flash-attn --no-build-isolation
if %errorlevel% neq 0 (
    echo ERROR: Failed to install flash-attn.
    exit /b 1
)

REM Install pygame (for audio playback in GUI)
echo Installing pygame ...
pip install -U pygame
if %errorlevel% neq 0 (
    echo ERROR: Failed to install pygame.
    exit /b 1
)

REM Install PySide6 (Qt GUI)
echo Installing PySide6-Essentials ...
pip install -U PySide6-Essentials
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PySide6.
    exit /b 1
)

echo.
echo Installation complete.
echo.
echo To run the GUI:
echo   conda activate %ENV_NAME%
echo   python %SCRIPT_NAME%
echo.
pause
