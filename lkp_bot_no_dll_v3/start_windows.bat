@echo off
setlocal
cd /d "%~dp0"
if not exist ".env" (
  echo Missing .env. Run setup_windows.bat first.
  pause
  exit /b 1
)
py -3 --version >nul 2>&1
if errorlevel 1 (
  echo Python 3 was not found. Install Python and enable PATH.
  pause
  exit /b 1
)
py -3 -c "import telegram,dotenv,httpx" >nul 2>&1
if errorlevel 1 (
  echo Installing Python requirements...
  py -3 -m pip install -r "requirements.txt"
  if errorlevel 1 (
    echo Python package installation failed.
    pause
    exit /b 1
  )
)
echo Starting Telegram bot. Leave this window open.
py -3 "bot.py"
echo.
echo Bot stopped or failed to start. Read the error above.
pause
