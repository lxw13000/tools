#!/bin/bash

# 图片分析系统 - 停止脚本

echo "=========================================="
echo "  图片分析系统 - 停止服务"
echo "=========================================="
echo ""

# 检测 docker compose 命令（v2 优先，v1 兜底）
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE="docker-compose"
else
    echo "错误: 未检测到 Docker Compose"
    exit 1
fi

$COMPOSE down

if [ $? -eq 0 ]; then
    echo "服务已停止"
else
    echo "停止服务失败"
    exit 1
fi
