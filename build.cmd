@echo off
rem ===========================================================================
rem  build.cmd -- one-command rebuild of AutoPC (onedir PyInstaller build).
rem
rem  Just run:   build        (or double-click it, or  .\build.cmd )
rem
rem  Output and PyInstaller's scratch go to a configurable location, and TEMP is
rem  redirected there too so a full system drive cannot break the build. A fresh
rem  .env.local is seeded next to the exe from .env.template (the --noconfirm
rem  build wipes dist first). Your own .env.local is never copied, so a build
rem  folder you share cannot leak your settings.
rem
rem  The built exe is a CONSOLE app: run it for the interactive shell, or pass a
rem  subcommand (AutoPC.exe run "...", AutoPC.exe skills, AutoPC.exe -h).
rem ===========================================================================
setlocal
set "ROOT=%~dp0"
rem Build output and PyInstaller scratch default to folders beside this
rem script; point them off any cloud-synced drive by setting AUTOPC_VENV,
rem AUTOPC_DIST and AUTOPC_WORK in your environment first.
if not defined AUTOPC_VENV set "AUTOPC_VENV=%ROOT%.venv"
if not defined AUTOPC_DIST set "AUTOPC_DIST=%ROOT%dist"
if not defined AUTOPC_WORK set "AUTOPC_WORK=%ROOT%build"
set "PY=%AUTOPC_VENV%\Scripts\python.exe"
set "DIST=%AUTOPC_DIST%"
set "WORK=%AUTOPC_WORK%"
set "APPNAME=AutoPC"

if not exist "%PY%" (
    echo Could not find the AutoPC venv at %PY%.
    echo Create it, then:  %PY% -m pip install -e "%ROOT%.[dev]"
    exit /b 1
)

if not exist "%WORK%\tmp" mkdir "%WORK%\tmp"
set "TEMP=%WORK%\tmp"
set "TMP=%WORK%\tmp"

rem A running instance locks %APPNAME%.exe and makes PyInstaller fail with
rem "Access is denied". Stop it (and the SDK CLI it spawned) first.
taskkill /IM "%APPNAME%.exe" /F >nul 2>&1

echo Ensuring PyInstaller is installed ...
"%PY%" -m pip install --quiet pyinstaller || (echo pip install pyinstaller failed & exit /b 1)

echo Building %APPNAME% ...
"%PY%" -m PyInstaller "%ROOT%autopc.spec" --noconfirm --distpath "%DIST%" --workpath "%WORK%"
if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

copy /Y "%ROOT%.env.template" "%DIST%\%APPNAME%\.env.local" >nul

echo.
echo Done -^> %DIST%\%APPNAME%\%APPNAME%.exe
endlocal
