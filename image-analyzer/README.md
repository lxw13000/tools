# 图片分析系统

基于 Flask 的图片分析服务，提供动态检测和 NSFW 内容识别，支持 3 个检测模型和多模型融合决策。

## 功能特性

### 动态检测
- 上传多张连续图片，通过感知哈希算法判断是静态还是动态
- 返回逐帧相似度和平均相似度

### NSFW 内容检测

支持 3 个检测模型：

| 模型 | ID | 大小 | 输出 | 特点 |
|------|----|------|------|------|
| OpenNSFW2 (Yahoo) | `opennsfw2` | ~23MB | 二分类 | 最轻量，自动下载 |
| MobileNet V2 140 | `mobilenet` | ~17MB | 5 分类 | 最快，唯一可输出内容分类 |
| Falconsai ViT | `falconsai` | ~330MB | 二分类 | 最准确 (98%+) |

### 双分类标签体系

所有检测结果输出两组标签：

- **安全分类** (`safety`): 色情、暴力、正常 — 百分比
- **内容分类** (`content_type`): 人物、动漫、风景 — 百分比（仅 MobileNet 可输出，其他模型为 `null`）

### 多模型融合

融合模式同时调用多个模型，加权计算综合安全评分。支持 3 种决策策略：
- `weighted_average` (默认): 加权平均 + 保守拦截
- `any_block`: 任一模型拦截即拦截
- `majority`: 多数投票

### 处理动作

| 动作 | 说明 |
|------|------|
| `block` (拦截) | 直接拦截/删除 |
| `review` (复审) | 进入人工复审队列 |
| `pass` (放行) | 内容安全 |

## 项目结构

```
image-analyzer/
├── app/
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── motion_detector.py       # 动态检测
│   │   ├── nsfw_detector.py          # NSFW 检测门面 (3 模型路由)
│   │   ├── opennsfw2_detector.py     # OpenNSFW2 检测器
│   │   ├── falconsai_detector.py     # Falconsai ViT 检测器
│   │   └── fusion_detector.py        # 多模型融合引擎
│   ├── templates/
│   │   ├── index.html                # 首页
│   │   ├── motion.html               # 动态检测页
│   │   └── nsfw.html                 # NSFW 检测页
│   └── app.py                        # Flask 主应用
├── models/                           # 模型文件目录
├── config.yaml                       # 配置文件
├── requirements.txt
├── download_models.py                # 模型下载脚本
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 快速开始

### Docker 部署（推荐）

```bash
# 1. 下载模型
python download_models.py

# 2. 构建并启动
docker-compose up -d --build

# 3. 访问服务
open http://localhost:5000
```

### 本地开发

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 安装 CPU-only PyTorch（节省空间）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. 下载模型
python download_models.py

# 4. 运行
python -m flask --app app.app run --host 0.0.0.0 --port 5000
```

生产环境使用 gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 2 --timeout 120 --preload app.app:app
```

## API

### 健康检查

```bash
GET /api/health
```

```json
{"status": "ok", "message": "服务运行正常"}
```

### 获取模型列表

```bash
GET /api/nsfw/models
```

返回 3 个模型的信息（名称、精度、速度、大小、是否可用）。

### 获取默认阈值

```bash
GET /api/nsfw/config
```

返回 MobileNet 5-class 的默认阈值配置。

### 动态检测

```bash
POST /api/detect/motion
# form-data: images=@frame1.jpg images=@frame2.jpg images=@frame3.jpg
```

```json
{
  "status": "success",
  "result": "static",
  "confidence": 0.98,
  "similarities": [0.99, 0.97],
  "message": "平均相似度: 98.00%"
}
```

### 单模型 NSFW 检测

```bash
POST /api/detect/nsfw
# form-data: image=@test.jpg model_id=mobilenet
```

MobileNet 响应（5 分类 → 双分类标签）：

```json
{
  "status": "success",
  "model": "MobileNet V2 140",
  "model_id": "mobilenet",
  "raw_scores": {
    "drawings": 0.05, "hentai": 0.02,
    "neutral": 0.85, "porn": 0.03, "sexy": 0.05
  },
  "content_type": {"人物": 0.93, "动漫": 0.02, "风景": 0.05},
  "safety": {"色情": 0.10, "暴力": 0.0, "正常": 0.90},
  "action": "pass",
  "action_text": "放行",
  "elapsed_seconds": 0.35
}
```

OpenNSFW2 / Falconsai 响应（二分类）：

```json
{
  "status": "success",
  "model": "OpenNSFW2 (Yahoo)",
  "model_id": "opennsfw2",
  "raw_scores": {"sfw": 0.92, "nsfw": 0.08},
  "content_type": null,
  "safety": {"色情": 0.08, "暴力": 0.0, "正常": 0.92},
  "action": "pass",
  "action_text": "放行",
  "elapsed_seconds": 0.28
}
```

### 融合检测

```bash
POST /api/detect/nsfw/fusion
# form-data: image=@test.jpg models=opennsfw2,mobilenet,falconsai
```

```json
{
  "status": "success",
  "fusion": {
    "final_score": 0.0823,
    "action": "pass",
    "action_text": "放行",
    "strategy": "weighted_average",
    "model_scores": {"opennsfw2": 0.08, "mobilenet": 0.10, "falconsai": 0.07},
    "details": ["opennsfw2: 8.00%", "mobilenet: 10.00%", "falconsai: 7.00%", "融合分数: 8.23%"]
  },
  "content_type": {"人物": 0.93, "动漫": 0.02, "风景": 0.05},
  "safety": {"色情": 0.0823, "暴力": 0.0, "正常": 0.9177},
  "model_results": {"opennsfw2": {...}, "mobilenet": {...}, "falconsai": {...}},
  "elapsed_seconds": 1.52
}
```

## 配置

配置文件 `config.yaml`:

```yaml
nsfw_detection:
  # MobileNet 5-class 阈值
  thresholds:
    porn: 0.6           # > 此值 → 拦截
    hentai: 0.5         # > 此值 → 复审
    sexy: 0.7           # > 此值 → 复审
    porn_hentai: 0.65   # 组合 > 此值 → 拦截

  # OpenNSFW2 阈值
  opennsfw2:
    thresholds:
      nsfw_block: 0.8
      nsfw_review: 0.5

  # Falconsai 阈值
  falconsai:
    thresholds:
      nsfw_block: 0.8
      nsfw_review: 0.5

  # 融合配置
  fusion:
    weights:
      opennsfw2: 0.25
      mobilenet: 0.30
      falconsai: 0.45
    thresholds:
      block: 0.7
      review: 0.4
    strategy: "weighted_average"
```

修改后重启: `docker-compose restart`

## 技术栈

- **后端**: Flask 3.0 + Gunicorn
- **模型框架**: TensorFlow 2.15 + PyTorch (CPU) + HuggingFace Transformers
- **图片处理**: Pillow, OpenCV
- **动态检测**: imagehash
- **容器化**: Docker + Docker Compose

## 许可证

本项目仅供学习和研究使用。
