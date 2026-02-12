@echo off
chcp 65001 >nul
echo ============================================
echo   VisionAce - EXE Build Script
echo ============================================
echo.

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller not found. Installing...
    pip install pyinstaller
    echo.
)

echo [1/3] Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo       Done.
echo.

echo [2/3] Building EXE with PyInstaller...
pyinstaller visionace.spec
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed! Check the output above.
    pause
    exit /b 1
)
echo.

echo [3/3] Build complete!
echo.
echo   Output: dist\VisionAce\VisionAce.exe
echo.
echo ============================================
pause
