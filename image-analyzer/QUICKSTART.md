# 快速开始指南

## 一键启动

### Linux/Mac 用户

```bash
cd image-analyzer
./start.sh
```

### Windows 用户

```bash
cd image-analyzer
docker-compose up -d --build
```

## 访问服务

启动成功后，在浏览器中打开：http://localhost:5000

## 停止服务

### Linux/Mac 用户

```bash
./stop.sh
```

### Windows 用户

```bash
docker-compose down
```

## 测试 API

### 1. 健康检查

```bash
curl http://localhost:5000/api/health
```

### 2. 动态检测（需要准备多张图片）

```bash
curl -X POST http://localhost:5000/api/detect/motion \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg"
```

### 3. NSFW 检测（需要准备一张图片）

```bash
curl -X POST http://localhost:5000/api/detect/nsfw \
  -F "image=@test.jpg"
```

## 修改配置

编辑 `config.yaml` 文件，修改阈值后重启服务：

```bash
docker-compose restart
```

## 查看日志

```bash
docker-compose logs -f
```

## 常见问题

1. **端口被占用**: 修改 `docker-compose.yml` 中的端口映射
2. **服务启动慢**: 首次启动需要下载 OpenNSFW2 模型，请耐心等待
3. **内存不足**: 减少 Dockerfile 中的 worker 数量
