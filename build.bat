@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo VisionAce Improved - Windows x64 CPU portable build
echo Use Python 3.11 x64 in a dedicated virtual environment.
echo Install requirements-cpu.txt, requirements.txt and requirements-build.txt first.
echo This command does not install packages or delete existing output directories.
if exist "dist\VisionAce-Improved" (
    echo Output already exists. Preserve or rename dist\VisionAce-Improved before building again.
    exit /b 1
)
set YOLO_AUTOINSTALL=false
set YOLO_OFFLINE=true
set MPLBACKEND=Agg
python -m PyInstaller --clean visionace.spec
if errorlevel 1 exit /b 1
echo Build complete: dist\VisionAce-Improved\VisionAce-Improved.exe
echo Distribute the entire VisionAce-Improved folder. The EXE alone is not portable.
endlocal
