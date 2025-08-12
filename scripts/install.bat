@echo off
echo Setting up development environment...

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Set up GDAL environment
python scripts\setup_env.py

REM Install Python packages
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo To start developing:
echo   1. Run: venv\Scripts\activate
echo   2. Run: python scripts\setup_env.py
echo   3. Start your application
pause