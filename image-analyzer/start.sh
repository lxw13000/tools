#!/bin/bash

# 图片分析系统 - 快速启动脚本

set -e

echo "=========================================="
echo "  图片分析系统 - 启动脚本"
echo "=========================================="
echo ""

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: 未检测到 Docker，请先安装 Docker"
    exit 1
fi

# 检测 docker compose 命令（v2 优先，v1 兜底）
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "错误: 未检测到 Docker Compose，请先安装"
    exit 1
fi
echo "Docker 环境检查通过 (使用 $COMPOSE)"
echo ""

# 创建运行时目录
mkdir -p uploads logs models
echo "创建 uploads/logs/models 目录"
echo ""

# 构建并启动服务
echo "开始构建 Docker 镜像..."
$COMPOSE build

echo "镜像构建成功"
echo ""
echo "启动服务..."
$COMPOSE up -d

echo "服务启动成功！"
echo ""
echo "=========================================="
echo "  访问地址"
echo "=========================================="
echo "Web 测试界面:     http://localhost:5000"
echo "健康检查:         http://localhost:5000/api/health"
echo "动态检测 API:     http://localhost:5000/api/detect/motion"
echo "NSFW 检测 API:    http://localhost:5000/api/detect/nsfw"
echo "NSFW 服务 API:    http://localhost:5000/api/detect/nsfw/check"
echo ""
echo "=========================================="
echo "  常用命令"
echo "=========================================="
echo "查看日志: $COMPOSE logs -f"
echo "停止服务: $COMPOSE down"
echo "重启服务: $COMPOSE restart"
echo "查看状态: $COMPOSE ps"
echo ""
echo "等待服务完全启动..."
sleep 5

# 轮询健康检查
for i in {1..12}; do
    if curl -sf http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "服务已就绪！可以开始使用了"
        exit 0
    fi
    echo "等待服务启动... ($i/12)"
    sleep 5
done

echo "警告: 服务启动超时，请手动检查: $COMPOSE logs"
exit 1
