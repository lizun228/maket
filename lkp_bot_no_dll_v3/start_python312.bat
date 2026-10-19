@echo off
setlocal
cd /d "%~dp0"
title LKP Bot - Python 3.12

echo ========================================
echo     LKP BOT - CLEAN PYTHON 3.12
echo ========================================

if not exist "bot.py" (
  echo ERROR: bot.py is missing.
  echo Move this BAT file into your existing lkp_bot_no_dll_v3 folder.
  goto :failed
)
if not exist "requirements.txt" (
  echo ERROR: requirements.txt is missing.
  goto :failed
)
if not exist ".env" (
  echo ERROR: .env is missing. Run setup_windows.bat first.
  goto :failed
)

py -3.12 -c "import sys; from typing import TypeVarTuple, Unpack; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Working Python 3.12 was not found.
  echo Install Python 3.12.10 from https://www.python.org/downloads/release/python-31210/
  echo Then run this BAT file again.
  goto :failed
)

echo Checking a clean Python 3.12 environment...
if not exist ".venv312\Scripts\python.exe" (
  echo Creating isolated environment...
  py -3.12 -m venv ".venv312"
  if errorlevel 1 goto :failed
)
if not exist ".venv312\.deps_installed" (
  echo Installing dependencies into isolated environment...
  ".venv312\Scripts\python.exe" -m pip install --disable-pip-version-check -r "requirements.txt"
  if errorlevel 1 goto :failed
  type nul > ".venv312\.deps_installed"
)

".venv312\Scripts\python.exe" -c "from typing import TypeVarTuple, Unpack; import telegram, dotenv, httpx; print('Dependencies OK')"
if errorlevel 1 goto :failed

echo.
echo Starting Telegram bot. Leave this window open.
".venv312\Scripts\python.exe" "bot.py"
echo.
echo Bot stopped. See messages above.
goto :end

:failed
echo.
echo Setup failed. See messages above.
:end
pause
endlocal
