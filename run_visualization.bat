@echo off
echo Running Python installation check...
call find_python.bat
if %errorlevel% neq 0 (
    echo Failed to find Python installation
    pause
    exit /b 1
)

echo.
echo Running visualization script...
%PYTHON_CMD% visualize_results.py
if %errorlevel% neq 0 (
    echo Failed to run visualization script
    pause
    exit /b 1
)

echo.
echo Visualization complete! Check MedicalData/results/ for the generated plots.
pause 