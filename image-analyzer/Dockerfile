# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
# opencv-python需要的系统库 + curl(healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      libgl1 \
      curl \
      && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 CPU-only PyTorch（从 PyTorch 官方 CPU 源，体积更小）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 安装其余 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ /app/app/
COPY config.yaml /app/

# 创建模型目录（模型文件通过 volume 挂载）
RUN mkdir -p /app/models

# 创建上传目录
RUN mkdir -p /tmp/uploads

# 暴露端口
EXPOSE 5000

# 健康检查（使用 curl，无需额外 Python 包）
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# 使用gunicorn运行应用（生产环境）
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "--preload", "app.app:app"]
