@echo off
rem FrameForensics launcher (Windows): double-click to run.
rem First run creates frameforensics-backend\.venv and installs dependencies (a few minutes).
setlocal
cd /d "%~dp0frameforensics-backend"

set "VENV_PY=.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto :deps

echo Looking for Python 3.10 - 3.13 ...
set "PYCMD="
for %%V in (3.12 3.11 3.10 3.13) do (
    if not defined PYCMD (
        py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo.
    echo [ERROR] Python 3.10 - 3.13 was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup, then run this file again.
    goto :fail
)

echo Creating virtual environment with: %PYCMD%
%PYCMD% -m venv .venv || goto :fail

:deps
rem Reinstall only when requirements-app.txt changed since the last install.
fc /b requirements-app.txt .venv\installed-requirements.txt >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies, this can take a few minutes the first time ...
    "%VENV_PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
    "%VENV_PY%" -m pip install -r requirements-app.txt --disable-pip-version-check || goto :fail
    copy /y requirements-app.txt .venv\installed-requirements.txt >nul
)

echo.
echo Starting FrameForensics at http://127.0.0.1:7860  (close this window to stop)
echo.
"%VENV_PY%" app\app.py || goto :fail
goto :eof

:fail
echo.
echo FrameForensics did not start. See the messages above.
pause
exit /b 1
