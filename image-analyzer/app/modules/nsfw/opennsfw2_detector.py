"""
OpenNSFW2 检测模块

基于 Yahoo OpenNSFW 的 TensorFlow 2 移植版本，
ResNet-50-thin 架构，模型大小 ~23MB，输出二分类 NSFW 概率。

线程安全设计：
  - 模型加载使用 Lock 保护，防止多线程重复初始化
  - 模型推理使用 Lock 串行化，TF predict() 非线程安全
  - 加载失败后标记熔断，避免反复重试消耗资源
"""

import os
import time
import logging
import threading
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OpenNSFW2Detector:
    """OpenNSFW2 (Yahoo) 二分类 NSFW 检测器"""

    def __init__(self, weights_path: str = '/app/models/open_nsfw_weights.h5',
                 config: Dict = None):
        self.weights_path = weights_path
        self._model = None          # 模型懒加载
        self._load_failed = False   # 熔断标记
        self._lock = threading.Lock()

        # 默认阈值
        self.thresholds = {
            'nsfw_block': 0.8,
            'nsfw_review': 0.5,
        }
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('opennsfw2', {}).get('thresholds', {})
            for key in ['nsfw_block', 'nsfw_review']:
                if key in t:
                    self.thresholds[key] = float(t[key])

        logger.info("OpenNSFW2: 模型地址 = %s", self.weights_path)

    @staticmethod
    def check_files(weights_path: str) -> bool:
        """静态方法：检查权重文件是否存在（不触发实例化）"""
        return os.path.isfile(weights_path)

    def is_available(self) -> bool:
        """检查 opennsfw2 依赖与权重文件是否就绪"""
        try:
            import opennsfw2  # noqa: F401
        except ImportError:
            return False
        return self.check_files(self.weights_path)

    def _ensure_loaded(self):
        """确保模型已加载（线程安全，含熔断保护）

        权重文件不存在时直接熔断，绝不联网下载。
        opennsfw2 库默认在文件缺失时会从 GitHub 下载，此处通过显式传 weights_path
        并提前校验文件存在性，规避离线/受限网络环境下的下载失败。
        """
        if self._model is not None:
            return
        if self._load_failed:
            raise RuntimeError("OpenNSFW2 模型加载曾失败，已熔断，请检查依赖与权重文件后重启服务")

        with self._lock:
            if self._model is not None:
                return
            try:
                if not os.path.isfile(self.weights_path):
                    raise FileNotFoundError(
                        f"OpenNSFW2 权重文件不存在: {self.weights_path}，"
                        f"请下载 open_nsfw_weights.h5 至该路径"
                    )
                logger.info("OpenNSFW2: 开始加载模型")
                load_start = time.perf_counter()
                import opennsfw2 as n2
                self._model = n2.make_open_nsfw_model(weights_path=self.weights_path)
                load_ms = int(round((time.perf_counter() - load_start) * 1000))
                logger.info("OpenNSFW2: 模型加载完成，耗时 %dms", load_ms)
            except Exception:
                self._load_failed = True
                logger.exception("OpenNSFW2: 模型加载失败，已标记熔断")
                raise

    def detect(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        """
        二分类 NSFW 检测

        Returns:
            dict: 统一安全标签格式（二分类模型不返回性感字段）
        """
        if not os.path.exists(image_path):
            logger.warning("OpenNSFW2: 图片文件不存在 %s", image_path)
            return {"status": "error", "message": "图片文件不存在"}

        t = thresholds if thresholds else self.thresholds

        try:
            start = time.perf_counter()
            self._ensure_loaded()

            import opennsfw2 as n2
            from PIL import Image

            # 图片预处理（Yahoo 官方预处理流程）
            with Image.open(image_path) as pil_image:
                pil_rgb = pil_image.convert('RGB')

            try:
                image = n2.preprocess_image(pil_rgb, n2.Preprocessing.YAHOO)
            finally:
                pil_rgb.close()

            inputs = np.expand_dims(image, axis=0)

            # 模型推理（Lock 串行化）
            with self._lock:
                predictions = self._model.predict(inputs, verbose=0)

            nsfw_prob = round(float(predictions[0][1]), 4)
            normal_prob = round(float(predictions[0][0]), 4)

            file_size = os.path.getsize(image_path)
            elapsed_seconds_raw = time.perf_counter() - start
            elapsed = round(elapsed_seconds_raw, 2)
            elapsed_ms = int(round(elapsed_seconds_raw * 1000))

            # 阈值判定
            block_th = t.get('nsfw_block', 0.8)
            review_th = t.get('nsfw_review', 0.5)

            if nsfw_prob >= block_th:
                action, action_text = 'block', '拦截'
                details = [f"NSFW {nsfw_prob:.2%} >= 拦截阈值 {block_th:.2%}"]
            elif nsfw_prob >= review_th:
                action, action_text = 'review', '复审'
                details = [f"NSFW {nsfw_prob:.2%} >= 复审阈值 {review_th:.2%}"]
            else:
                action, action_text = 'pass', '放行'
                details = []

            logger.info("OpenNSFW2: action=%s, nsfw=%.4f, elapsed=%dms (%.2fs)",
                        action, nsfw_prob, elapsed_ms, elapsed)

            return {
                'status': 'success',
                'model': 'OpenNSFW2 (Yahoo)',
                'model_id': 'opennsfw2',
                'elapsed_seconds': elapsed,
                'elapsed_ms': elapsed_ms,
                'image_size': file_size,
                'raw_scores': {'nsfw': nsfw_prob, 'normal': normal_prob},
                'content_type': None,
                'safety': {
                    '色情': nsfw_prob,
                    '正常': normal_prob,
                },
                'action': action,
                'action_text': action_text,
                'details': details,
            }

        except Exception as e:
            logger.exception("OpenNSFW2 检测失败")
            return {"status": "error", "message": f"OpenNSFW2 检测失败: {str(e)}"}
