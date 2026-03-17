# 内容安全检测（鉴黄）功能说明

## 一、功能概述

本系统提供基于深度学习的图片内容安全检测（鉴黄）能力，支持三种 AI 模型独立检测和多模型融合检测，输出三级判定结果：**拦截(block) / 复审(review) / 放行(pass)**。

检测能力已封装为标准 REST API 服务，支持两种调用方式：

| 方式 | 接口 | 输入 | 适用场景 |
|------|------|------|----------|
| 文件上传 | `POST /api/detect/nsfw` | multipart form 上传图片文件 | 前端页面、本地文件检测 |
| URL 远程调用 | `POST /api/detect/nsfw/check` | JSON 传入图片 URL | 外部业务系统集成 |

---

## 二、核心架构

```
                        ┌──────────────────────────────┐
                        │       API 入口层 (app.py)     │
                        │  /api/detect/nsfw             │
                        │  /api/detect/nsfw/fusion      │
                        │  /api/detect/nsfw/check       │
                        └──────────┬───────────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                  │
        ┌────────▼───────┐ ┌──────▼──────┐  ┌───────▼───────┐
        │  NsfwService   │ │ NSFWDetector │  │FusionDetector │
        │ (URL下载+限流) │ │  (门面路由)  │  │ (加权融合)    │
        └────────┬───────┘ └──────┬──────┘  └───────┬───────┘
                 │                │                  │
                 └────────────────┼──────────────────┘
                                  │
               ┌──────────────────┼──────────────────┐
               │                  │                   │
      ┌────────▼────────┐ ┌──────▼──────┐ ┌──────────▼──────────┐
      │  OpenNSFW2      │ │  MobileNet  │ │    Falconsai ViT    │
      │  (Yahoo)        │ │  V2 140     │ │                     │
      │  ResNet-50      │ │  5分类      │ │  Vision Transformer │
      │  二分类 ~23MB   │ │  ~17MB      │ │  二分类 ~330MB      │
      └─────────────────┘ └─────────────┘ └─────────────────────┘
```

### 层级说明

- **API 入口层**：接收 HTTP 请求，参数校验，文件管理，异常捕获
- **NsfwService**：面向外部业务的服务层，负责 URL 图片下载、信号量限流、全链路异常隔离
- **NSFWDetector（门面）**：统一管理三个模型实例的生命周期（懒加载），路由 `detect()` 调用到对应模型
- **FusionDetector**：多模型加权融合引擎，收集各模型分数，按策略输出综合判定
- **三个模型检测器**：各自负责图片预处理、模型推理、阈值判定

---

## 三、模型详情

### 3.1 模型对比

| 属性 | MobileNet V2 140 | OpenNSFW2 (Yahoo) | Falconsai ViT |
|------|-------------------|-------------------|---------------|
| 架构 | MobileNet V2 | ResNet-50-thin | ViT-base-patch16-224 |
| 模型大小 | ~17MB | ~23MB | ~330MB |
| 推理速度 | 最快 | 快 | 较慢 |
| 精度 | 较高 | 较高 | 最高 (98%+) |
| 输出类型 | 5 分类 | 二分类 | 二分类 |
| 内容分类 | 支持（人物/动漫/风景） | 不支持 | 不支持 |
| 性感检测 | 支持（独立字段） | 不支持 | 不支持 |
| 框架 | TensorFlow/Keras | TensorFlow 2 | PyTorch (HuggingFace) |

### 3.2 MobileNet V2 140（5 分类模型）

唯一支持细粒度分类的模型。模型输出 5 个类别概率：

| 原始类别 | 含义 |
|----------|------|
| `drawings` | 绘画/风景 |
| `hentai` | 动漫色情 |
| `neutral` | 中性/安全 |
| `porn` | 色情 |
| `sexy` | 性感 |

**标签映射规则：**

```
安全分类 (safety):
  色情 = porn + hentai
  性感 = sexy
  正常 = neutral + drawings

内容分类 (content_type):
  人物 = porn + sexy + neutral
  动漫 = hentai
  风景 = drawings
```

**级联判定逻辑**（优先级从高到低，命中即停止）：

```
1. porn > porn阈值(0.6)            → 拦截 (block)
2. porn + hentai > 组合阈值(0.65)  → 拦截 (block)
3. hentai > hentai阈值(0.5)        → 复审 (review)
4. sexy > sexy阈值(0.7)            → 复审 (review)
5. 均未触发                         → 放行 (pass)
```

### 3.3 OpenNSFW2 / Falconsai ViT（二分类模型）

输出 `nsfw` 和 `normal` 两个概率值，判定逻辑：

```
nsfw_score >= nsfw_block(0.8)   → 拦截 (block)
nsfw_score >= nsfw_review(0.5)  → 复审 (review)
nsfw_score < nsfw_review        → 放行 (pass)
```

> 二分类模型不输出「性感」字段和「内容分类」，避免产生误导性的零值。

### 3.4 多模型融合检测

融合引擎调用多个模型，通过加权平均计算综合分数：

```
final_porn = Σ(模型色情分数 × 模型权重) / Σ(成功模型权重)
final_sexy = Σ(模型性感分数 × 模型权重) / Σ(提供性感分数的模型权重)   # 仅 MobileNet 贡献
combined_score = final_porn + final_sexy
```

**默认模型权重：**

| 模型 | 权重 | 原因 |
|------|------|------|
| Falconsai ViT | 0.45 | 精度最高，主导融合结果 |
| MobileNet V2 | 0.30 | 速度最快，提供内容分类补充 |
| OpenNSFW2 | 0.25 | 辅助交叉验证 |

**三种决策策略：**

| 策略 | 逻辑 |
|------|------|
| `weighted_average`（默认，保守） | 综合分数超阈值 **或** 任一模型拦截 → 拦截 |
| `any_block` | 任一模型拦截 → 拦截；否则用综合分数判定 |
| `majority` | 多数模型投票：过半拦截 → 拦截；过半拦截+复审 → 复审 |

---

## 四、参数策略配置

### 4.1 配置文件（config.yaml）

```yaml
# NSFW 检测配置
nsfw_detection:
  # MobileNet V2 140 (5-class) 阈值
  thresholds:
    porn: 0.6           # 色情 > 此值 → 拦截
    hentai: 0.5         # 动漫色情 > 此值 → 复审
    sexy: 0.7           # 性感 > 此值 → 复审
    porn_hentai: 0.65   # 色情+动漫 组合 > 此值 → 拦截

  # OpenNSFW2 (Yahoo) 阈值
  opennsfw2:
    thresholds:
      nsfw_block: 0.8   # NSFW >= 此值 → 拦截
      nsfw_review: 0.5  # NSFW >= 此值 → 复审

  # Falconsai ViT 阈值
  falconsai:
    thresholds:
      nsfw_block: 0.8   # NSFW >= 此值 → 拦截
      nsfw_review: 0.5  # NSFW >= 此值 → 复审

  # 多模型融合配置
  fusion:
    weights:
      opennsfw2: 0.25
      mobilenet: 0.30
      falconsai: 0.45
    thresholds:
      block: 0.7        # 融合分数 >= 此值 → 拦截
      review: 0.4       # 融合分数 >= 此值 → 复审
    strategy: "weighted_average"

# NSFW 检测服务配置（供外部业务调用）
nsfw_service:
  max_concurrent: 5       # 最大并发检测数（信号量限流）
  queue_timeout: 30       # 排队等待超时（秒），超时返回 503
  download_timeout: 15    # 图片下载超时（秒）
  max_file_size: 52428800 # 图片最大大小 50MB
```

### 4.2 阈值总览

| 模型 | 阈值参数 | 默认值 | 触发动作 |
|------|----------|--------|----------|
| MobileNet | `porn` | 0.6 | block |
| MobileNet | `porn_hentai` | 0.65 | block |
| MobileNet | `hentai` | 0.5 | review |
| MobileNet | `sexy` | 0.7 | review |
| OpenNSFW2 | `nsfw_block` | 0.8 | block |
| OpenNSFW2 | `nsfw_review` | 0.5 | review |
| Falconsai | `nsfw_block` | 0.8 | block |
| Falconsai | `nsfw_review` | 0.5 | review |
| 融合 | `block` | 0.7 | block |
| 融合 | `review` | 0.4 | review |

> 所有阈值均支持 API 调用时动态覆盖，不传则使用配置文件默认值。

---

## 五、接口说明

### 5.1 外部业务调用接口（推荐）

供其他业务系统集成的核心接口，接受图片 URL，自行下载并检测。

```
POST /api/detect/nsfw/check
Content-Type: application/json
```

**请求参数：**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `imgUrl` | string | 是 | 图片网络 URL 地址 |
| `modelStrategy` | object | 否 | 模型参数策略，为空时使用 Falconsai ViT 默认参数 |
| `modelStrategy.modelId` | string | 否 | 模型ID：`falconsai` / `opennsfw2` / `mobilenet` / `fusion`，默认 `falconsai` |
| `modelStrategy.models` | array | 否 | 融合模式下参与的模型列表，默认全部三个 |
| `modelStrategy.strategy` | string | 否 | 融合策略：`weighted_average` / `any_block` / `majority` |
| `modelStrategy.thresholds` | object | 否 | 阈值参数（单模型直传，融合模式按模型ID分组） |

**响应状态码：**

| 状态码 | 含义 |
|--------|------|
| 200 | 检测成功 |
| 400 | 参数错误或图片下载失败 |
| 500 | 服务内部错误 |
| 503 | 服务繁忙（并发排队超时） |

### 5.2 文件上传检测接口

```
POST /api/detect/nsfw
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 图片文件 |
| `model_id` | string | 否 | 模型ID，默认 `mobilenet` |
| `threshold_*` | float | 否 | 阈值覆盖（如 `threshold_porn=0.5`） |

### 5.3 融合检测接口

```
POST /api/detect/nsfw/fusion
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image` | file | 是 | 图片文件 |
| `models` | string | 否 | 逗号分隔的模型ID列表 |
| `threshold_*` | float | 否 | 阈值覆盖 |

### 5.4 辅助接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/nsfw/models` | GET | 返回所有模型信息及可用性 |
| `/api/nsfw/config` | GET | 返回 MobileNet 默认阈值 |
| `/api/health` | GET | 健康检查 |

---

## 六、接口调用示例

### 6.1 最简调用（默认 Falconsai ViT）

```bash
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg"
  }'
```

`modelStrategy` 为空，自动使用 Falconsai ViT 模型 + 默认阈值（nsfw_block=0.8, nsfw_review=0.5）。

### 6.2 指定 Falconsai + 自定义阈值

```bash
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg",
    "modelStrategy": {
      "modelId": "falconsai",
      "thresholds": {
        "nsfw_block": 0.7,
        "nsfw_review": 0.4
      }
    }
  }'
```

### 6.3 使用 MobileNet 5 分类模型

```bash
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg",
    "modelStrategy": {
      "modelId": "mobilenet",
      "thresholds": {
        "porn": 0.5,
        "hentai": 0.4,
        "sexy": 0.6,
        "porn_hentai": 0.55
      }
    }
  }'
```

### 6.4 使用 OpenNSFW2 模型

```bash
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg",
    "modelStrategy": {
      "modelId": "opennsfw2",
      "thresholds": {
        "nsfw_block": 0.75,
        "nsfw_review": 0.45
      }
    }
  }'
```

### 6.5 多模型融合检测（全部模型）

```bash
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg",
    "modelStrategy": {
      "modelId": "fusion",
      "models": ["opennsfw2", "mobilenet", "falconsai"],
      "strategy": "weighted_average",
      "thresholds": {
        "mobilenet": {
          "porn": 0.6,
          "hentai": 0.5,
          "sexy": 0.7,
          "porn_hentai": 0.65
        },
        "opennsfw2": {
          "nsfw_block": 0.8,
          "nsfw_review": 0.5
        },
        "falconsai": {
          "nsfw_block": 0.8,
          "nsfw_review": 0.5
        }
      }
    }
  }'
```

### 6.6 融合检测（仅两个模型 + 投票策略）

```bash
curl -X POST http://localhost:5000/api/detect/nsfw/check \
  -H "Content-Type: application/json" \
  -d '{
    "imgUrl": "https://example.com/photo.jpg",
    "modelStrategy": {
      "modelId": "fusion",
      "models": ["falconsai", "opennsfw2"],
      "strategy": "any_block"
    }
  }'
```

不传 `thresholds`，各模型使用配置文件默认阈值。

---

## 七、响应格式

### 7.1 单模型检测结果（二分类：Falconsai / OpenNSFW2）

```json
{
  "status": "success",
  "model": "Falconsai ViT",
  "model_id": "falconsai",
  "elapsed_seconds": 0.28,
  "image_size": 123456,
  "raw_scores": {
    "nsfw": 0.0823,
    "normal": 0.9177
  },
  "content_type": null,
  "safety": {
    "色情": 0.0823,
    "正常": 0.9177
  },
  "action": "pass",
  "action_text": "放行",
  "details": [],
  "request_id": "a1b2c3d4e5f6",
  "imgUrl": "https://example.com/photo.jpg",
  "total_elapsed_seconds": 1.52
}
```

### 7.2 单模型检测结果（5 分类：MobileNet）

```json
{
  "status": "success",
  "model": "MobileNet V2 140",
  "model_id": "mobilenet",
  "elapsed_seconds": 0.35,
  "image_size": 123456,
  "raw_scores": {
    "drawings": 0.0500,
    "hentai": 0.0200,
    "neutral": 0.8500,
    "porn": 0.0300,
    "sexy": 0.0500
  },
  "content_type": {
    "人物": 0.9300,
    "动漫": 0.0200,
    "风景": 0.0500
  },
  "safety": {
    "色情": 0.0500,
    "性感": 0.0500,
    "正常": 0.9000
  },
  "action": "pass",
  "action_text": "放行",
  "details": [],
  "request_id": "a1b2c3d4e5f6",
  "imgUrl": "https://example.com/photo.jpg",
  "total_elapsed_seconds": 1.85
}
```

### 7.3 多模型融合检测结果

```json
{
  "status": "success",
  "fusion": {
    "final_score": 0.0823,
    "final_porn": 0.0600,
    "final_sexy": 0.0223,
    "action": "pass",
    "action_text": "放行",
    "strategy": "weighted_average",
    "model_scores": {
      "opennsfw2": 0.08,
      "mobilenet": 0.05,
      "falconsai": 0.06
    },
    "details": [
      "opennsfw2: 色情 8.00%",
      "mobilenet: 色情 5.00%",
      "falconsai: 色情 6.00%",
      "融合色情: 6.00%",
      "融合性感: 2.23%",
      "综合分数: 8.23%"
    ]
  },
  "content_type": {
    "人物": 0.9300,
    "动漫": 0.0200,
    "风景": 0.0500
  },
  "safety": {
    "色情": 0.0600,
    "性感": 0.0223,
    "正常": 0.9177
  },
  "model_results": {
    "opennsfw2": { "...单模型完整结果..." },
    "mobilenet": { "...单模型完整结果..." },
    "falconsai": { "...单模型完整结果..." }
  },
  "elapsed_seconds": 2.15,
  "request_id": "a1b2c3d4e5f6",
  "imgUrl": "https://example.com/photo.jpg",
  "total_elapsed_seconds": 3.42
}
```

### 7.4 错误响应

```json
{
  "status": "error",
  "message": "图片下载失败，请检查 URL 是否可访问",
  "request_id": "a1b2c3d4e5f6",
  "imgUrl": "https://example.com/broken.jpg",
  "elapsed_seconds": 2.01
}
```

---

## 八、高并发与稳定性设计

### 8.1 请求处理流程

```
请求进入
  │
  ├─ 参数校验（imgUrl 非空、modelStrategy JSON 合法）
  │    └─ 失败 → 400
  │
  ├─ 信号量限流 acquire(timeout=30s)
  │    └─ 超时 → 503 "服务繁忙"
  │
  ├─ 下载图片（timeout=15s，限 50MB，流式分块写入）
  │    └─ 失败 → 400 "图片下载失败"
  │
  ├─ 执行模型检测
  │    └─ 异常 → 500 "检测服务内部错误"
  │
  └─ finally: 释放信号量 + 删除临时文件（始终执行）
```

### 8.2 保护机制

| 机制 | 配置 | 说明 |
|------|------|------|
| 信号量限流 | `max_concurrent: 5` | 最多 5 个检测任务同时执行，防止 CPU/内存耗尽 |
| 排队超时 | `queue_timeout: 30` | 等待超 30s 直接返回 503，防止请求堆积雪崩 |
| 下载超时 | `download_timeout: 15` | 15s 超时，防止慢连接阻塞工作线程 |
| 文件大小限制 | `max_file_size: 50MB` | 流式分块检查，超限立即中止并删除 |
| 全链路异常隔离 | try/except 每环节 | 下载、检测、清理各自独立捕获，互不影响 |
| 临时文件强制清理 | finally 块 | 无论成功失败，始终删除临时文件 |
| 请求 ID 追踪 | `request_id` | 12 位 UUID，贯穿全部日志输出，便于排查 |

### 8.3 模型懒加载

三个模型均采用懒加载策略：首次调用时才初始化，避免启动时一次性加载全部模型占用过多内存。加载后常驻内存，后续调用无需重复加载。

---

## 九、可视化测试页面

访问 `/nsfw-service` 可打开内容安全检测服务测试页面，功能包括：

- **URL 输入框**：输入图片网络地址
- **模型策略编辑器**：JSON 格式编辑，提供 5 个快捷模板一键填充
  - Falconsai 默认 / OpenNSFW2 / MobileNet / 融合模式 / 留空
- **检测执行**：发送请求并展示结果
- **结果可视化**：action 状态标签、安全分类条形图、融合评分、各模型子结果
- **原始 JSON**：请求参数和响应 JSON 可折叠展示，便于调试
