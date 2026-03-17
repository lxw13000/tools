@echo off
chcp 65001 >nul 2>nul

echo ==========================================
echo   图片分析系统 - 停止服务 (Windows)
echo ==========================================
echo.

docker compose down

if %errorlevel%==0 (
    echo 服务已停止
) else (
    echo 停止服务失败
    exit /b 1
)
