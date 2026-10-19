@echo off
setlocal
cd /d "%~dp0"
echo Stop the bot before backing up. Keep the backup private.
py -3.12 backup_project.py
if errorlevel 1 echo Backup failed. Do not update until the backup succeeds.
pause
