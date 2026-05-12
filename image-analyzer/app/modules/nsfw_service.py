"""
NSFW 内容安全检测服务

封装鉴黄检测能力为独立服务，供外部业务通过 API 调用。
接受图片 URL，自行下载图片并执行检测，返回统一格式的结果 JSON。

高并发与稳定性设计：
  - 信号量限流：限制同时执行的检测任务数量，防止 CPU/内存耗尽
  - 排队超时：信号量等待超时直接返回 503，防止请求堆积雪崩
  - 图片下载超时 + 大小限制：防止慢连接和超大文件阻塞
  - URL 安全校验：仅允许 http/https，目标 IP 必须命中白名单（纯白名单模式）
  - 下载后图片格式验证：确保文件为有效图片再交给模型
  - 全链路 try/except：每个环节独立捕获异常，不影响其他业务
  - 临时文件强制清理：finally 块确保磁盘不泄漏
  - 请求 ID 日志追踪：贯穿整条调用链路，便于排查问题
"""

import os
import uuid
import time
import socket
import ipaddress
import logging
import threading
import requests as http_requests
from urllib.parse import urlparse, unquote
from PIL import Image

logger = logging.getLogger(__name__)

# 允许的 URL 协议
_ALLOWED_SCHEMES = {'http', 'https'}


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
        # 默认模型 ID：请求未传 modelStrategy 或缺 modelId 时的兜底（见 _parse_strategy）
        self.default_model_id = svc_config.get('default_model_id', 'falconsai')

        # IP 白名单网段（纯白名单模式）：仅允许图片 URL 解析到的目标 IP 命中此处配置的网段
        # 未配置或全部解析失败 → 一律拒绝（见 _is_ip_allowed）
        # 解析失败的条目记 warning 后跳过，不阻断启动
        self.ip_whitelist = []
        for net_str in svc_config.get('ip_whitelist', []) or []:
            try:
                self.ip_whitelist.append(ipaddress.ip_network(net_str, strict=False))
            except (ValueError, TypeError) as e:
                logger.warning("ip_whitelist 配置项无效 '%s': %s", net_str, e)

        # 临时文件存放目录
        self.upload_folder = self.config.get('upload', {}).get('folder', '/tmp/uploads')
        os.makedirs(self.upload_folder, exist_ok=True)

        # 信号量限流：限制同时执行的检测任务数
        self._semaphore = threading.Semaphore(self.max_concurrent)

        logger.info(
            "NsfwService 初始化完成, default_model_id=%s, max_concurrent=%d, "
            "queue_timeout=%ds, download_timeout=%ds, max_file_size=%dMB, "
            "ip_whitelist=%s",
            self.default_model_id, self.max_concurrent, self.queue_timeout,
            self.download_timeout, self.max_file_size // (1024 * 1024),
            [str(n) for n in self.ip_whitelist] or '无（将拒绝所有图片 URL）',
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

        # 解析模型策略（为空则使用 default_model_id 配置项）
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
            filepath, download_err = self._download_image(img_url, request_id)
            if filepath is None:
                elapsed = round(time.time() - start_time, 2)
                return {
                    "status": "error",
                    "message": download_err or "图片下载失败",
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
            logger.info("[%s] 未提供模型策略，使用默认模型: %s",
                        request_id, self.default_model_id)
            return self.default_model_id, None, None, None

        model_id = model_strategy.get('modelId', self.default_model_id)
        models = model_strategy.get('models')
        strategy = model_strategy.get('strategy')
        thresholds = model_strategy.get('thresholds')

        logger.info("[%s] 解析策略: modelId=%s, models=%s, strategy=%s",
                    request_id, model_id, models, strategy)
        return model_id, models, strategy, thresholds

    @staticmethod
    def _is_ip_allowed(hostname, allowed_nets):
        """
        检查主机名解析到的 IP 是否全部命中白名单网段（纯白名单模式）

        所有解析结果必须都在白名单内才放行；任一 IP 不在白名单 → 拒绝。
        白名单为空、DNS 返回空、DNS 解析失败 → 一律拒绝（无法验证即拒绝）。

        Args:
            hostname:     URL 中的主机名
            allowed_nets: 白名单网段列表（ip_network 对象）

        Returns:
            bool: True 为放行，False 为拒绝
        """
        if not allowed_nets:
            return False  # 未配置白名单 → 一律拒绝

        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
            if not addr_info:
                return False  # DNS 返回空 → 拒绝

            for _, _, _, _, sockaddr in addr_info:
                ip = ipaddress.ip_address(sockaddr[0])
                if not any(ip in net for net in allowed_nets):
                    return False  # 任一解析 IP 不在白名单 → 拒绝
            return True
        except (socket.gaierror, ValueError):
            return False  # DNS 解析失败或地址非法 → 拒绝

    def _validate_url(self, img_url, request_id):
        """
        校验 URL 合法性：协议白名单 + 目标 IP 白名单（纯白名单模式）

        Returns:
            (parsed, error_msg) — parsed 为 urlparse 结果，校验失败时 parsed=None
        """
        parsed = urlparse(img_url)

        if parsed.scheme not in _ALLOWED_SCHEMES:
            logger.warning("[%s] URL 协议不允许: %s", request_id, parsed.scheme)
            return None, f"仅支持 http/https 协议，当前: {parsed.scheme}"

        hostname = parsed.hostname
        if not hostname:
            logger.warning("[%s] URL 缺少主机名: %s", request_id, img_url)
            return None, "URL 格式无效：缺少主机名"

        if not self._is_ip_allowed(hostname, self.ip_whitelist):
            logger.warning("[%s] URL 目标 IP 不在白名单中被拦截: %s", request_id, hostname)
            return None, "图片 URL 目标 IP 不在白名单中"

        return parsed, None

    def _validate_image(self, filepath, request_id):
        """
        验证下载的文件是否为有效图片

        Returns:
            bool: True 为有效图片
        """
        try:
            with Image.open(filepath) as img:
                img.verify()  # 快速校验图片头部，不解码全部数据
            return True
        except Exception:
            logger.warning("[%s] 文件不是有效图片: %s", request_id, filepath)
            return False

    def _download_image(self, img_url, request_id):
        """
        从 URL 下载图片到本地临时目录

        安全措施：URL 协议白名单、私有 IP 拦截、连接+读取双超时、
        流式分块写入带大小限制、下载后验证图片格式有效性。

        Args:
            img_url:    图片 URL
            request_id: 请求追踪 ID

        Returns:
            (str, str): (文件路径, None) 成功; (None, 错误原因) 失败
        """
        filepath = None
        try:
            # URL 安全校验
            parsed, err = self._validate_url(img_url, request_id)
            if parsed is None:
                return None, err

            logger.info("[%s] 开始下载图片: %s", request_id, img_url)

            # 提取文件名
            path = unquote(parsed.path)
            filename = os.path.basename(path) if path else 'image.jpg'
            if '.' not in filename:
                filename += '.jpg'

            # 流式下载：(connect_timeout, read_timeout) 双超时
            # 使用常见浏览器 UA，避免被 CDN/对象存储拦截
            with http_requests.get(
                img_url,
                timeout=(self.download_timeout, self.download_timeout * 2),
                stream=True,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                                  'Chrome/120.0.0.0 Safari/537.36',
                },
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()

                # 预检 Content-Length
                content_length = resp.headers.get('Content-Length')
                if content_length and int(content_length) > self.max_file_size:
                    logger.warning("[%s] 图片过大: %s bytes, 上限 %d",
                                   request_id, content_length, self.max_file_size)
                    return None, f"图片过大（{int(content_length) // (1024*1024)}MB），上限 {self.max_file_size // (1024*1024)}MB"

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
                                break
                            f.write(chunk)
                    else:
                        # for 循环正常结束（未 break），下载完成
                        pass

                # 超出大小限制时清理文件并返回
                if downloaded_size > self.max_file_size:
                    self._safe_remove(filepath, request_id)
                    return None, f"下载超出大小限制 {self.max_file_size // (1024*1024)}MB"

            # 下载后验证文件是否为有效图片
            if not self._validate_image(filepath, request_id):
                self._safe_remove(filepath, request_id)
                return None, "下载的文件不是有效图片"

            logger.info("[%s] 图片下载完成: %s, 大小=%d bytes",
                        request_id, filepath, downloaded_size)
            return filepath, None

        except http_requests.exceptions.Timeout:
            logger.warning("[%s] 图片下载超时 (%ds): %s",
                           request_id, self.download_timeout, img_url)
            self._safe_remove(filepath, request_id)
            return None, f"图片下载超时（{self.download_timeout}s）"
        except http_requests.exceptions.ConnectionError as e:
            logger.warning("[%s] 图片下载连接失败: %s, 错误: %s",
                           request_id, img_url, str(e))
            self._safe_remove(filepath, request_id)
            return None, "图片下载连接失败，请检查 URL 域名是否可访问"
        except http_requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 'N/A'
            logger.warning("[%s] 图片下载 HTTP 错误 %s: %s",
                           request_id, status_code, img_url)
            self._safe_remove(filepath, request_id)
            return None, f"图片服务器返回 HTTP {status_code} 错误"
        except http_requests.exceptions.RequestException as e:
            logger.warning("[%s] 图片下载失败: %s, 错误: %s",
                           request_id, img_url, str(e))
            self._safe_remove(filepath, request_id)
            return None, f"图片下载失败: {str(e)}"
        except Exception:
            logger.exception("[%s] 图片下载发生未预期异常: %s", request_id, img_url)
            self._safe_remove(filepath, request_id)
            return None, "图片下载发生未预期异常"

    @staticmethod
    def _safe_remove(filepath, request_id):
        """安全删除文件（忽略不存在或权限错误）"""
        if filepath:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                logger.debug("[%s] 临时文件清理失败: %s", request_id, filepath)

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
