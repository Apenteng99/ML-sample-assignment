@echo off
echo Running Python installation check...
call find_python.bat
if %errorlevel% neq 0 (
    echo Failed to find Python installation
    pause
    exit /b 1
)

echo.
echo Installing required Python packages...
%PYTHON_CMD% -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Failed to upgrade pip
    pause
    exit /b 1
)

echo Installing packages from requirements.txt...
%PYTHON_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Failed to install required packages
    pause
    exit /b 1
)

echo.
echo Verifying installations...
%PYTHON_CMD% -c "import matplotlib; import seaborn; import pandas; import sklearn; import numpy; print('All packages installed successfully!')"
if %errorlevel% neq 0 (
    echo Some packages failed to install correctly
    pause
    exit /b 1
)

echo.
echo Installation complete! All packages are installed and working correctly.
pause 