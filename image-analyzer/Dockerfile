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

# 安装 CPU-only PyTorch（从官方 CPU 源，体积更小）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
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
