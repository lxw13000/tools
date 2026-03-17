"""
NSFW 内容安全检测服务

封装鉴黄检测能力为独立服务，供外部业务通过 API 调用。
接受图片 URL，自行下载图片并执行检测，返回统一格式的结果 JSON。

高并发与稳定性设计：
  - 信号量限流：限制同时执行的检测任务数量，防止 CPU/内存耗尽
  - 排队超时：信号量等待超时直接返回 503，防止请求堆积雪崩
  - 图片下载超时 + 大小限制：防止慢连接和超大文件阻塞
  - 全链路 try/except：每个环节独立捕获异常，不影响其他业务
  - 临时文件强制清理：finally 块确保磁盘不泄漏
  - 请求 ID 日志追踪：贯穿整条调用链路，便于排查问题
"""

import os
import uuid
import time
import logging
import threading
import requests as http_requests
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)


class NsfwService:
    """NSFW 内容安全检测服务：接收图片 URL，下载并执行检测"""

    def __init__(self, nsfw_detector, fusion_detector, config=None):
        """
        初始化检测服务

        Args:
            nsfw_detector:   NSFWDetector 门面实例
            fusion_detector: FusionDetector 融合检测器实例
            config:          全局配置字典（来自 config.yaml）
        """
        self.nsfw_detector = nsfw_detector
        self.fusion_detector = fusion_detector
        self.config = config or {}

        # 读取服务配置
        svc_config = self.config.get('nsfw_service', {})
        self.max_concurrent = int(svc_config.get('max_concurrent', 5))
        self.queue_timeout = int(svc_config.get('queue_timeout', 30))
        self.download_timeout = int(svc_config.get('download_timeout', 15))
        self.max_file_size = int(svc_config.get('max_file_size', 52428800))  # 50MB

        # 临时文件存放目录
        self.upload_folder = self.config.get('upload', {}).get('folder', '/tmp/uploads')
        os.makedirs(self.upload_folder, exist_ok=True)

        # 信号量限流：限制同时执行的检测任务数
        self._semaphore = threading.Semaphore(self.max_concurrent)

        logger.info(
            "NsfwService 初始化完成, max_concurrent=%d, queue_timeout=%ds, "
            "download_timeout=%ds, max_file_size=%dMB",
            self.max_concurrent, self.queue_timeout,
            self.download_timeout, self.max_file_size // (1024 * 1024),
        )

    def check(self, img_url, model_strategy=None):
        """
        执行内容安全检测（对外核心方法）

        Args:
            img_url:        图片网络 URL（必填）
            model_strategy: 模型参数策略字典（可选），格式：
                {
                    "modelId": "falconsai",           # 模型 ID 或 "fusion"
                    "models": ["opennsfw2", "falconsai"],  # 融合模式下参与的模型
                    "strategy": "weighted_average",    # 融合策略
                    "thresholds": {...}                # 阈值参数
                }

        Returns:
            dict: 检测结果 JSON
        """
        request_id = uuid.uuid4().hex[:12]
        start_time = time.time()

        logger.info("[%s] 收到检测请求, imgUrl=%s, modelStrategy=%s",
                    request_id, img_url, model_strategy)

        # ---- 参数校验 ----
        if not img_url or not isinstance(img_url, str) or not img_url.strip():
            logger.warning("[%s] imgUrl 参数为空", request_id)
            return {"status": "error", "message": "imgUrl 不能为空"}, 400

        img_url = img_url.strip()

        # 解析模型策略（为空则默认 Falconsai ViT）
        model_id, models, strategy, thresholds = self._parse_strategy(
            model_strategy, request_id
        )

        # ---- 信号量限流 ----
        acquired = self._semaphore.acquire(timeout=self.queue_timeout)
        if not acquired:
            elapsed = round(time.time() - start_time, 2)
            logger.warning("[%s] 服务繁忙，排队超时 %ds, elapsed=%.2fs",
                           request_id, self.queue_timeout, elapsed)
            return {
                "status": "error",
                "message": f"服务繁忙，请稍后重试（排队超时 {self.queue_timeout}s）",
                "request_id": request_id,
                "elapsed_seconds": elapsed,
            }, 503

        filepath = None
        try:
            # ---- 下载图片 ----
            filepath = self._download_image(img_url, request_id)
            if filepath is None:
                elapsed = round(time.time() - start_time, 2)
                return {
                    "status": "error",
                    "message": "图片下载失败，请检查 URL 是否可访问",
                    "request_id": request_id,
                    "imgUrl": img_url,
                    "elapsed_seconds": elapsed,
                }, 400

            # ---- 执行检测 ----
            if model_id == 'fusion':
                result = self._detect_fusion(
                    filepath, models, thresholds, request_id
                )
            else:
                result = self._detect_single(
                    filepath, model_id, thresholds, request_id
                )

            elapsed = round(time.time() - start_time, 2)

            # 补充服务层信息
            result['request_id'] = request_id
            result['imgUrl'] = img_url
            result['total_elapsed_seconds'] = elapsed

            logger.info("[%s] 检测完成, action=%s, elapsed=%.2fs",
                        request_id, result.get('action', result.get('fusion', {}).get('action', 'N/A')),
                        elapsed)

            return result, 200

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            logger.exception("[%s] 检测过程发生未预期异常", request_id)
            return {
                "status": "error",
                "message": f"检测服务内部错误: {str(e)}",
                "request_id": request_id,
                "imgUrl": img_url,
                "elapsed_seconds": elapsed,
            }, 500

        finally:
            # 释放信号量
            self._semaphore.release()
            # 清理临时文件
            if filepath:
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                        logger.debug("[%s] 临时文件已清理: %s", request_id, filepath)
                except OSError as e:
                    logger.warning("[%s] 临时文件清理失败 %s: %s",
                                   request_id, filepath, e)

    def _parse_strategy(self, model_strategy, request_id):
        """
        解析模型参数策略

        Returns:
            (model_id, models, strategy, thresholds) 四元组
        """
        if not model_strategy or not isinstance(model_strategy, dict):
            logger.info("[%s] 未提供模型策略，使用默认 Falconsai ViT", request_id)
            return 'falconsai', None, None, None

        model_id = model_strategy.get('modelId', 'falconsai')
        models = model_strategy.get('models')
        strategy = model_strategy.get('strategy')
        thresholds = model_strategy.get('thresholds')

        logger.info("[%s] 解析策略: modelId=%s, models=%s, strategy=%s",
                    request_id, model_id, models, strategy)
        return model_id, models, strategy, thresholds

    def _download_image(self, img_url, request_id):
        """
        从 URL 下载图片到本地临时目录

        Args:
            img_url:    图片 URL
            request_id: 请求追踪 ID

        Returns:
            str: 下载后的本地文件路径，失败返回 None
        """
        try:
            logger.info("[%s] 开始下载图片: %s", request_id, img_url)

            # 提取文件名
            parsed = urlparse(img_url)
            path = unquote(parsed.path)
            filename = os.path.basename(path) if path else 'image.jpg'
            # 确保文件名有扩展名
            if '.' not in filename:
                filename += '.jpg'

            # 流式下载，支持超时和大小限制
            resp = http_requests.get(
                img_url,
                timeout=self.download_timeout,
                stream=True,
                headers={'User-Agent': 'ImageAnalyzer/1.0'},
            )
            resp.raise_for_status()

            # 检查 Content-Length（如果服务端提供）
            content_length = resp.headers.get('Content-Length')
            if content_length and int(content_length) > self.max_file_size:
                logger.warning("[%s] 图片过大: %s bytes, 上限 %d",
                               request_id, content_length, self.max_file_size)
                resp.close()
                return None

            # 写入临时文件（分块读取，边下边检查大小）
            unique_filename = f"svc_{request_id}_{filename}"
            filepath = os.path.join(self.upload_folder, unique_filename)
            downloaded_size = 0

            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_file_size:
                            logger.warning("[%s] 下载超出大小限制 %d bytes",
                                           request_id, self.max_file_size)
                            f.close()
                            os.remove(filepath)
                            return None
                        f.write(chunk)

            logger.info("[%s] 图片下载完成: %s, 大小=%d bytes",
                        request_id, filepath, downloaded_size)
            return filepath

        except http_requests.exceptions.Timeout:
            logger.warning("[%s] 图片下载超时 (%ds): %s",
                           request_id, self.download_timeout, img_url)
            return None
        except http_requests.exceptions.RequestException as e:
            logger.warning("[%s] 图片下载失败: %s, 错误: %s",
                           request_id, img_url, str(e))
            return None
        except Exception as e:
            logger.exception("[%s] 图片下载发生未预期异常: %s", request_id, img_url)
            return None

    def _detect_single(self, filepath, model_id, thresholds, request_id):
        """
        单模型检测

        Args:
            filepath:   本地图片路径
            model_id:   模型 ID
            thresholds: 阈值参数字典
            request_id: 请求追踪 ID
        """
        try:
            logger.info("[%s] 执行单模型检测: model=%s", request_id, model_id)
            result = self.nsfw_detector.detect(
                filepath,
                model_id=model_id,
                thresholds=thresholds,
            )
            return result
        except Exception as e:
            logger.exception("[%s] 单模型检测异常: model=%s", request_id, model_id)
            return {
                "status": "error",
                "message": f"模型检测失败: {str(e)}",
                "model_id": model_id,
            }

    def _detect_fusion(self, filepath, models, thresholds, request_id):
        """
        多模型融合检测

        Args:
            filepath:   本地图片路径
            models:     参与融合的模型 ID 列表
            thresholds: 按模型 ID 分组的阈值字典
            request_id: 请求追踪 ID
        """
        try:
            logger.info("[%s] 执行融合检测: models=%s", request_id, models)
            result = self.fusion_detector.detect(
                filepath,
                models=models,
                thresholds=thresholds,
            )
            return result
        except Exception as e:
            logger.exception("[%s] 融合检测异常", request_id)
            return {
                "status": "error",
                "message": f"融合检测失败: {str(e)}",
            }
