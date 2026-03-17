# 图片分析系统

基于 Flask 的图片分析服务，提供动态检测和 NSFW 内容安全检测，支持 3 个检测模型和多模型融合决策。

## 功能特性

### 图片动态检测
- 上传多张连续图片（1-6 张），融合三种算法判断静态/动态
- 算法：感知哈希 (pHash) + 结构相似度 (SSIM) + 光流检测 (Optical Flow)
- 输出三级判定：静态 / 动态 / 人工复核
- 详见 [MOTION_DETECTION.md](MOTION_DETECTION.md)

### NSFW 内容安全检测（鉴黄）
- 支持 3 个检测模型，可单模型检测或多模型融合
- 输出三级判定：拦截(block) / 复审(review) / 放行(pass)
- 对外提供标准 REST API，支持 URL 远程调用
- 详见 [NSFW_DETECTION.md](NSFW_DETECTION.md)

| 模型 | ID | 大小 | 输出 | 特点 |
|------|----|------|------|------|
| MobileNet V2 140 | `mobilenet` | ~17MB | 5 分类 | 最快，唯一支持内容分类和性感检测 |
| OpenNSFW2 (Yahoo) | `opennsfw2` | ~23MB | 二分类 | 轻量级，自动下载 |
| Falconsai ViT | `falconsai` | ~330MB | 二分类 | 最准确 (98%+) |

### 定时检测服务
- 定时从外部 API 拉取待检测数据，自动批量执行动态检测
- 支持结果回调通知
- 可视化管理页面

## 项目结构

```
image-analyzer/
├── app/
│   ├── app.py                        # Flask 主应用（路由、初始化）
│   ├── logging_config.py             # 统一日志配置（轮转、分级）
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── motion_detector.py        # 动态检测（pHash+SSIM+光流融合）
│   │   ├── nsfw_detector.py          # NSFW 检测门面（3 模型路由）
│   │   ├── opennsfw2_detector.py     # OpenNSFW2 检测器
│   │   ├── falconsai_detector.py     # Falconsai ViT 检测器
│   │   ├── fusion_detector.py        # 多模型融合引擎
│   │   ├── nsfw_service.py           # NSFW 检测服务（URL下载+限流+SSRF防护）
│   │   └── scheduler_service.py      # 定时检测服务
│   └── templates/
│       ├── index.html                # 首页导航
│       ├── motion.html               # 动态检测测试页
│       ├── nsfw.html                 # NSFW 检测测试页
│       ├── nsfw_service.html         # NSFW 服务测试页
│       └── scheduler.html            # 定时服务管理页
├── models/                           # 模型文件目录（volume 挂载）
├── config.yaml                       # 全局配置文件
├── requirements.txt                  # Python 依赖
├── download_models.py                # 模型下载脚本
├── Dockerfile                        # 容器构建（非 root 用户）
├── docker-compose.yml                # 容器编排（资源限制）
├── start.sh / start.bat              # 一键启动脚本
├── stop.sh / stop.bat                # 一键停止脚本
├── NSFW_DETECTION.md                 # 鉴黄功能详细文档
├── MOTION_DETECTION.md               # 动态检测详细文档
├── QUICKSTART.md                     # 快速入门指南
└── README.md
```

## 快速开始

### Docker 部署（推荐）

```bash
# 1. 下载模型
python download_models.py

# 2. 一键启动（Linux/Mac）
chmod +x start.sh && ./start.sh

# 2. 一键启动（Windows）
start.bat

# 3. 访问服务
# http://localhost:5000
```

或手动操作：

```bash
docker compose build
docker compose up -d
```

### 本地开发

```bash
# 1. 安装 CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 安装其余依赖
pip install -r requirements.txt

# 3. 下载模型
python download_models.py

# 4. 运行
python -m flask --app app.app run --host 0.0.0.0 --port 5000
```

生产环境使用 gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 2 --timeout 120 --preload app.app:app
```

## API 接口一览

### 页面路由

| 路径 | 说明 |
|------|------|
| `/` | 首页导航 |
| `/motion` | 动态检测测试页 |
| `/nsfw` | NSFW 检测测试页 |
| `/nsfw-service` | NSFW 服务测试页 |
| `/scheduler` | 定时服务管理页 |

### REST API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/nsfw/models` | GET | 返回所有模型信息及可用性 |
| `/api/nsfw/config` | GET | 返回 MobileNet 默认阈值 |
| `/api/motion/config` | GET | 返回动态检测默认权重和阈值 |
| `/api/detect/motion` | POST | 图片序列动态检测（multipart 上传） |
| `/api/detect/nsfw` | POST | 单模型 NSFW 检测（multipart 上传） |
| `/api/detect/nsfw/fusion` | POST | 多模型融合检测（multipart 上传） |
| `/api/detect/nsfw/check` | POST | **外部业务调用接口**（JSON，传 URL） |
| `/api/scheduler/config` | GET | 定时服务配置信息 |
| `/api/scheduler/status` | GET | 定时服务运行状态 |
| `/api/scheduler/trigger` | POST | 手动触发一次批次执行 |

### 外部业务调用示例

```bash
# 最简调用（默认 Falconsai ViT）
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{"imgUrl": "https://example.com/photo.jpg"}'

# 指定模型 + 自定义阈值
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg",
    "modelStrategy": {
      "modelId": "falconsai",
      "thresholds": {"nsfw_block": 0.7, "nsfw_review": 0.4}
    }
  }'

# 多模型融合
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg",
    "modelStrategy": {
      "modelId": "fusion",
      "strategy": "weighted_average"
    }
  }'
```

更多示例见 [NSFW_DETECTION.md](NSFW_DETECTION.md)

## 配置

所有配置集中在 `config.yaml`，修改后重启生效：`docker compose restart`

主要配置项：
- `logging` — 日志级别、目录、轮转策略
- `motion_detection` — 动态检测权重和阈值
- `nsfw_detection` — 各模型阈值、融合策略和权重
- `nsfw_service` — 并发限制、下载超时、文件大小限制
- `scheduler` — 定时任务开关和拉取间隔

## 高并发与稳定性设计

- **信号量限流**：NSFW 服务最多 5 个并发检测（可配置），超限排队，30s 超时返回 503
- **线程安全推理**：所有模型推理通过 Lock 串行化，防止多线程并发 crash
- **模型懒加载 + 熔断**：首次调用时加载模型，加载失败后标记熔断不再重试
- **URL 安全校验**：仅允许 http/https 协议，拦截私有 IP 地址（防 SSRF）
- **图片下载保护**：连接+读取双超时、流式大小检查、下载后格式验证
- **日志轮转**：RotatingFileHandler 自动轮转，防止磁盘占满
- **Docker 资源限制**：内存 4G 上限，CPU 2 核上限
- **非 root 运行**：容器内使用专用 appuser 用户

## 技术栈

- **后端**: Flask 3.0 + Gunicorn
- **模型框架**: TensorFlow 2.15 + PyTorch (CPU) + HuggingFace Transformers
- **图片处理**: Pillow, OpenCV, imagehash, scikit-image
- **定时任务**: APScheduler
- **容器化**: Docker + Docker Compose

## 许可证

本项目仅供学习和研究使用。
