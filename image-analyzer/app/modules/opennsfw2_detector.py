"""
OpenNSFW2 检测模块

基于 Yahoo OpenNSFW 的 TensorFlow 2 移植版本，
使用 ResNet-50-thin 架构，模型大小 ~23MB，输出二分类 NSFW 概率。

二分类模型特性：
  - 只输出 nsfw / normal 两个概率值
  - 无法区分「色情」和「性感」，故安全分类中不返回性感字段（避免 0.0 造成理解偏差）
  - 内容分类（人物/动漫/风景）不支持，返回 None

阈值判定：
  - nsfw_prob >= nsfw_block  → 拦截
  - nsfw_prob >= nsfw_review → 复审
  - 其余                      → 放行
"""

import os
import time
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class OpenNSFW2Detector:
    """OpenNSFW2 (Yahoo) 二分类 NSFW 检测器"""

    def __init__(self, config: Dict = None):
        """
        初始化 OpenNSFW2 检测器

        Args:
            config: 全局配置字典，从中读取 nsfw_detection.opennsfw2.thresholds
        """
        self._model = None  # 模型懒加载

        # 默认阈值
        self.thresholds = {
            'nsfw_block': 0.8,   # NSFW >= 此值 → 拦截
            'nsfw_review': 0.5,  # NSFW >= 此值 → 复审
        }
        # 从配置文件覆盖默认阈值
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('opennsfw2', {}).get('thresholds', {})
            for key in ['nsfw_block', 'nsfw_review']:
                if key in t:
                    self.thresholds[key] = float(t[key])

        logger.info("OpenNSFW2Detector 初始化完成, thresholds=%s", self.thresholds)

    def is_available(self) -> bool:
        """检查 opennsfw2 依赖是否已安装"""
        try:
            import opennsfw2
            return True
        except ImportError:
            return False

    def _ensure_loaded(self):
        """确保模型已加载（懒加载：首次调用时通过 opennsfw2 创建模型）"""
        if self._model is not None:
            return
        logger.info("OpenNSFW2: 首次调用，开始加载模型")
        import opennsfw2 as n2
        self._model = n2.make_open_nsfw_model()
        logger.info("OpenNSFW2: 模型加载完成")

    def detect(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        """
        二分类 NSFW 检测

        Args:
            image_path: 图片文件绝对路径
            thresholds: 阈值覆盖字典 {nsfw_block, nsfw_review}

        Returns:
            dict: 统一安全标签格式（二分类模型不返回性感字段）
                成功: {status:'success', model, model_id, elapsed_seconds, image_size,
                       raw_scores, content_type(None), safety{色情,正常},
                       action, action_text, details}
                失败: {status:'error', message}
        """
        if not os.path.exists(image_path):
            logger.warning("OpenNSFW2: 图片文件不存在 %s", image_path)
            return {"status": "error", "message": "图片文件不存在"}

        # 使用传入阈值，若未传则用配置/默认阈值
        t = thresholds if thresholds else self.thresholds

        try:
            start = time.time()
            self._ensure_loaded()

            import opennsfw2 as n2
            from PIL import Image

            # ---- 图片预处理（使用 Yahoo 官方预处理流程） ----
            with Image.open(image_path) as pil_image:
                pil_rgb = pil_image.convert('RGB')
                image = n2.preprocess_image(pil_rgb, n2.Preprocessing.YAHOO)
            inputs = np.expand_dims(image, axis=0)  # 增加 batch 维度

            # ---- 模型推理 ----
            predictions = self._model.predict(inputs, verbose=0)
            # predictions[0] = [normal_prob, nsfw_prob]
            nsfw_prob = round(float(predictions[0][1]), 4)
            normal_prob = round(1.0 - nsfw_prob, 4)

            file_size = os.path.getsize(image_path)
            elapsed = round(time.time() - start, 2)

            # ---- 阈值判定 ----
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

            logger.info("OpenNSFW2: 检测完成, action=%s, nsfw=%.4f, normal=%.4f, "
                        "image_size=%d, elapsed=%.2fs",
                        action, nsfw_prob, normal_prob, file_size, elapsed)

            return {
                'status': 'success',
                'model': 'OpenNSFW2 (Yahoo)',
                'model_id': 'opennsfw2',
                'elapsed_seconds': elapsed,
                'image_size': file_size,
                'raw_scores': {'nsfw': nsfw_prob, 'normal': normal_prob},
                'content_type': None,  # 二分类模型不支持内容分类
                # 二分类模型仅输出色情/正常，无法区分性感，故不返回性感字段（避免 0.0 造成理解偏差）
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
