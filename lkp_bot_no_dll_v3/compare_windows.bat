@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv312\Scripts\python.exe" (
  echo Run start_python312.bat once to prepare Python dependencies.
  pause
  exit /b 1
)
echo Comparing old and new answers using local Ollama. This may take several minutes.
set PYTHONDONTWRITEBYTECODE=1
".venv312\Scripts\python.exe" evaluate.py --generate
if errorlevel 1 echo Comparison incomplete. Check Ollama and the installed model. No results were sent anywhere.
pause
