#!/bin/bash

# 图片分析系统 - 快速启动脚本

echo "=========================================="
echo "  图片分析系统 - 启动脚本"
echo "=========================================="
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未检测到 Docker，请先安装 Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: 未检测到 Docker Compose，请先安装 Docker Compose"
    exit 1
fi

echo "✅ Docker 环境检查通过"
echo ""

# 创建上传目录
mkdir -p uploads
# 创建日志目录
mkdir -p logs
echo "✅ 创建上传/日志目录"
echo ""

# 构建并启动服务
echo "🚀 开始构建 Docker 镜像..."
docker-compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker 镜像构建成功"
    echo ""
    echo "🚀 启动服务..."
    docker-compose up -d

    if [ $? -eq 0 ]; then
        echo "✅ 服务启动成功！"
        echo ""
        echo "=========================================="
        echo "  访问地址"
        echo "=========================================="
        echo "🌐 Web 测试界面: http://localhost:5000"
        echo "🔍 健康检查: http://localhost:5000/api/health"
        echo "📹 动态检测 API: http://localhost:5000/api/detect/motion"
        echo "🔞 NSFW 检测 API: http://localhost:5000/api/detect/nsfw"
        echo ""
        echo "=========================================="
        echo "  常用命令"
        echo "=========================================="
        echo "查看日志: docker-compose logs -f"
        echo "停止服务: docker-compose down"
        echo "重启服务: docker-compose restart"
        echo "查看状态: docker-compose ps"
        echo ""
        echo "⏳ 等待服务完全启动（约30秒）..."
        sleep 5

        # 检查服务是否就绪
        for i in {1..12}; do
            if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
                echo "✅ 服务已就绪！可以开始使用了"
                break
            fi
            echo "⏳ 等待服务启动... ($i/12)"
            sleep 5
        done
    else
        echo "❌ 服务启动失败，请查看日志: docker-compose logs"
        exit 1
    fi
else
    echo "❌ Docker 镜像构建失败"
    exit 1
fi
