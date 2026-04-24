@echo off
setlocal

REM Create and activate a local virtual environment.
if not exist .venv (
    py -m venv .venv
    if errorlevel 1 goto :error
)

call .venv\Scripts\activate
if errorlevel 1 goto :error

python -m pip install --upgrade pip
if errorlevel 1 goto :error

pip install -r requirements.txt
if errorlevel 1 goto :error

python -m playwright install chromium
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo.
echo To run the solver with bundled Chromium:
echo   python solve_web.py --browser chromium --size 8
echo.
echo To run with Microsoft Edge installed on your PC:
echo   python solve_web.py --browser msedge --size 8
echo.
goto :eof

:error
echo.
echo Setup failed. Review the error messages above.
exit /b 1
