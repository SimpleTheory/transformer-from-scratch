@echo off
setlocal

cd /d "%~dp0"

set "BRANCH_NAME=4090-run-v1"
set "SERVICE_NAME=trainer"

echo Checking repository...

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo.
    echo This folder is not a Git repository.
    echo Clone the repository first, then run this script again.
    pause
    exit /b 1
)

echo Updating project from branch %BRANCH_NAME%...
git pull --ff-only origin %BRANCH_NAME%

if errorlevel 1 (
    echo.
    echo Git pull failed.
    echo Please capture the error above and send it to the repository maintainer.
    pause
    exit /b 1
)

echo.
echo Building and starting the training container...
docker compose up -d --build %SERVICE_NAME%

if errorlevel 1 (
    echo.
    echo Docker failed to start the training container.
    echo Please capture the error above and send it to the repository maintainer.
    pause
    exit /b 1
)

echo.
echo Training started successfully.
echo Closing this window will not stop the container.
echo Press Ctrl+C to stop following the logs.
echo.

docker compose logs -f %SERVICE_NAME%

pause