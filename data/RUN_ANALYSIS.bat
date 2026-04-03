@echo off
REM Extract and Analyze CSV Files
REM Windows Batch Script - Double-click to run

cd /d C:\Users\prate\OneDrive\Desktop\legit0\data

echo.
echo ========================================
echo Data Analysis Script
echo ========================================
echo.

python extract_and_analyze.py

if errorlevel 1 (
    echo.
    echo Error running script!
    echo Make sure Python is installed and in your PATH
    pause
) else (
    echo.
    echo Script completed successfully!
    pause
)
