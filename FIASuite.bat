@echo off
rem  FIASuite launcher for a copy of the source code (git clone / zip download).
rem  First start: finds or installs Python 3.12, creates a private environment in this folder,
rem  installs the required packages, then starts the program. Later starts go straight to the program.
rem  (The portable FIASuite.exe build does not need this file.)
setlocal EnableExtensions
title FIASuite
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" goto :launch

echo.
echo   FIASuite - first start on this PC
echo   ---------------------------------
echo   A private Python environment is created inside this folder:
echo   %CD%\.venv
echo   Nothing is installed system-wide, except Python itself if it is missing.
echo.

call :find_python
if defined PY goto :have_python

echo   Python 3.12 was not found on this PC.
echo.
where winget >nul 2>&1
if errorlevel 1 goto :no_winget
choice /C YN /M "  Install Python 3.12 now with the Windows package manager (winget)"
if errorlevel 2 goto :manual_python
echo.
winget install --id Python.Python.3.12 -e --scope user
call :find_python
if defined PY goto :have_python
echo.
echo   Python was installed, but this window cannot see it yet.
echo   Close this window and double-click FIASuite.bat again.
goto :fail

:no_winget
echo   The Windows package manager (winget) is not available here.
:manual_python
echo.
echo   Please install Python 3.12 from  https://www.python.org/downloads/windows/
echo   (tick "Add python.exe to PATH" in the installer), then double-click FIASuite.bat again.
goto :fail

:have_python
echo   Using Python:  %PY%
echo   Creating the environment ...
%PY% -m venv .venv
if errorlevel 1 goto :venv_failed
echo   Downloading and installing the required packages (about 400 MB, this takes a few minutes) ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto :pip_failed
echo.
echo   Done. Starting FIASuite ...
echo   (ExifTool and the detection models are offered by the setup wizard inside the program.)

:launch
start "" ".venv\Scripts\pythonw.exe" -m fiasuite %*
exit /b 0

:venv_failed
echo   The environment could not be created.
goto :fail

:pip_failed
echo.
echo   The packages could not be installed. Check the internet connection and start FIASuite.bat again.
rmdir /s /q ".venv" >nul 2>&1
goto :fail

:fail
echo.
pause
exit /b 1

rem ---------------------------------------------------------------------------------------------
:find_python
set "PY="
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -3.12"
if defined PY goto :eof
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :eof
rem freshly installed by winget: not on this window's PATH yet
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
goto :eof
