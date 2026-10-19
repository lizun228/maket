@echo off
setlocal
cd /d "%~dp0"
if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
)
echo Edit .env: Telegram token, NUMERIC Telegram ID, Ollama model.
echo Never send your token to anyone.
notepad.exe ".env"
echo Saved? Run start_windows.bat.
pause
