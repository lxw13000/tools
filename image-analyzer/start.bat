@echo off
chcp 65001 >nul 2>nul

echo ==========================================
echo   图片分析系统 - 启动脚本 (Windows)
echo ==========================================
echo.

:: 检查 Docker
docker --version >nul 2>nul
if errorlevel 1 (
    echo 错误: 未检测到 Docker，请先安装 Docker Desktop
    exit /b 1
)

:: 检测 docker compose 命令
docker compose version >nul 2>nul
if errorlevel 1 (
    echo 错误: 未检测到 Docker Compose
    exit /b 1
)
echo Docker 环境检查通过
echo.

:: 创建运行时目录
if not exist uploads mkdir uploads
if not exist logs mkdir logs
if not exist models mkdir models
echo 创建 uploads/logs/models 目录
echo.

:: 构建并启动
echo 开始构建 Docker 镜像...
docker compose build
if errorlevel 1 (
    echo 镜像构建失败
    exit /b 1
)

echo 启动服务...
docker compose up -d
if errorlevel 1 (
    echo 服务启动失败
    exit /b 1
)

echo.
echo ==========================================
echo   访问地址
echo ==========================================
echo Web 测试界面:     http://localhost:5000
echo 健康检查:         http://localhost:5000/api/health
echo 动态检测 API:     http://localhost:5000/api/detect/motion
echo NSFW 检测 API:    http://localhost:5000/api/detect/nsfw
echo NSFW 服务 API:    http://localhost:5000/api/detect/nsfw/check
echo.
echo ==========================================
echo   常用命令
echo ==========================================
echo 查看日志: docker compose logs -f
echo 停止服务: docker compose down
echo 重启服务: docker compose restart
echo 查看状态: docker compose ps
echo.

echo 等待服务启动...
timeout /t 10 /nobreak >nul

curl -sf http://localhost:5000/api/health >nul 2>nul
if %errorlevel%==0 (
    echo 服务已就绪！可以开始使用了
) else (
    echo 服务可能尚在启动中，请稍后手动检查健康接口
)
