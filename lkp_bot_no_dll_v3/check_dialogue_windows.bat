@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
  echo Run start_python312.bat once to prepare Python dependencies.
  pause
  exit /b 1
)
echo Testing complete conversations with local Ollama. Reports remain on this PC.
set PYTHONDONTWRITEBYTECODE=1
".venv312\Scripts\python.exe" check_dialogue.py --generate
if errorlevel 1 echo Test incomplete. Check Ollama and the installed model.
pause
