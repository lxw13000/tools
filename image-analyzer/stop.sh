#!/bin/bash

# 图片分析系统 - 停止脚本

echo "=========================================="
echo "  图片分析系统 - 停止服务"
echo "=========================================="
echo ""

docker-compose down

if [ $? -eq 0 ]; then
    echo "✅ 服务已停止"
else
    echo "❌ 停止服务失败"
    exit 1
fi
