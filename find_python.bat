@echo off
echo Searching for Python installation...
where python > nul 2>&1
if %errorlevel% equ 0 (
    echo Found Python in PATH
    set PYTHON_CMD=python
) else (
    echo Python not found in PATH, searching in common locations...
    if exist "C:\Users\NANA SONNY 99\AppData\Local\Programs\Python\Python313\python.exe" (
        echo Found Python 3.13
        set PYTHON_CMD="C:\Users\NANA SONNY 99\AppData\Local\Programs\Python\Python313\python.exe"
    ) else if exist "C:\Python313\python.exe" (
        echo Found Python 3.13 in root
        set PYTHON_CMD="C:\Python313\python.exe"
    ) else (
        echo Python not found. Please install Python 3.13
        pause
        exit /b 1
    )
)
echo Using Python: %PYTHON_CMD%
echo.
echo Testing Python installation...
%PYTHON_CMD% --version
if %errorlevel% neq 0 (
    echo Error: Python installation not working correctly
    pause
    exit /b 1
)
echo.
echo Python installation verified successfully.
pause 