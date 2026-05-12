# 使用 Python 3.10 作为基础镜像
FROM python:3.10-slim

WORKDIR /app

# 环境变量：禁用输出缓冲、字节码、pip 缓存
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（opencv 所需库 + curl 用于 healthcheck）
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      libgl1 \
      curl \
      && rm -rf /var/lib/apt/lists/*

# 复制依赖清单
COPY requirements.txt .

# 安装 CPU-only PyTorch
# x86_64: 从官方 CPU 源安装（体积更小）
# arm64:  官方 CPU 源无 ARM64 wheel，退回 PyPI（Mac Apple Silicon / ARM 服务器）
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch; \
    fi

# 安装其余 Python 依赖（TensorFlow 等大包需要较长超时）
RUN pip install --no-cache-dir --timeout=300 --retries=3 -r requirements.txt

# 复制应用代码和配置
COPY app/ /app/app/
COPY config.yaml /app/

# 创建非 root 用户（生产安全最佳实践）
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# 创建运行时目录并设置权限
RUN mkdir -p /app/models /tmp/uploads /app/logs \
    && chown -R appuser:appuser /app /tmp/uploads

# 切换到非 root 用户
USER appuser

EXPOSE 5000

# 容器健康检查
# 参数说明：
#   --interval=300s     探测间隔（每 5 分钟探测 1 次）
#                       注意：每次探测都会在 gunicorn access log 中留下一行
#                       /api/health 记录，调大间隔可显著减少日志噪音。
#   --timeout=10s       单次探测超时上限，超过即视为本次失败
#   --start-period=60s  容器启动初期的宽限时间，期间失败不计入 retries
#                       （应用模型为懒加载，60s 足够 gunicorn + Flask 起来）
#   --retries=3         连续 3 次失败才标记容器为 unhealthy
#                       探测命令：curl 失败（非 2xx/超时/连接拒绝）即返回非 0
HEALTHCHECK --interval=300s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# 使用 gunicorn 运行（生产环境）
# --preload：预加载应用，减少每个 worker 的内存占用
# --access-logfile / --error-logfile：由 logging_config.py 中的 RotatingFileHandler 接管轮转
# 此处仍输出到 - (stdout) 供 docker logs 查看，文件落盘由应用层 handler 负责
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "2", \
     "--timeout", "120", \
     "--preload", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "app.app:app"]
