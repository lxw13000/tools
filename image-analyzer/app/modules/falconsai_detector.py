"""
Falconsai ViT NSFW 检测模块

基于 Falconsai/nsfw_image_detection 模型，ViT-base 架构，
模型大小 ~330MB，精度 98%+，输出二分类 nsfw / normal。

线程安全设计：
  - 模型加载使用 Lock 保护，防止多线程重复初始化
  - Pipeline 推理使用 Lock 串行化，HuggingFace pipeline 非线程安全
  - 加载失败后标记熔断，避免反复重试消耗资源
"""

import os
import time
import logging
import threading
from typing import Dict, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class FalconsaiDetector:
    """Falconsai ViT 二分类 NSFW 检测器"""

    def __init__(self, model_dir: str = '/app/models/falconsai', config: Dict = None):
        # 探测实际模型位置（snapshot_download 可能存在子目录）
        for sub in ['', 'nsfw_detection']:
            candidate = os.path.join(model_dir, sub) if sub else model_dir
            if os.path.exists(os.path.join(candidate, 'config.json')):
                self.model_dir = candidate
                break
        else:
            self.model_dir = model_dir

        self._pipeline = None       # HuggingFace pipeline 懒加载
        self._load_failed = False   # 熔断标记：加载失败后不再重试
        self._lock = threading.Lock()  # 保护加载和推理的线程安全

        # 默认阈值
        self.thresholds = {
            'nsfw_block': 0.8,
            'nsfw_review': 0.5,
        }
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('falconsai', {}).get('thresholds', {})
            for key in ['nsfw_block', 'nsfw_review']:
                if key in t:
                    self.thresholds[key] = float(t[key])

        logger.info("FalconsaiDetector 初始化完成, model_dir=%s", self.model_dir)

    @staticmethod
    def check_files(model_dir: str) -> bool:
        """静态方法：检查模型文件是否存在（不触发实例化）"""
        for sub in ['', 'nsfw_detection']:
            candidate = os.path.join(model_dir, sub) if sub else model_dir
            config_path = os.path.join(candidate, 'config.json')
            if os.path.exists(config_path):
                safetensors = os.path.join(candidate, 'model.safetensors')
                bin_path = os.path.join(candidate, 'pytorch_model.bin')
                return os.path.exists(safetensors) or os.path.exists(bin_path)
        return False

    def is_available(self) -> bool:
        """检查模型文件是否存在"""
        return self.check_files(os.path.dirname(self.model_dir)
                                if os.path.basename(self.model_dir) == 'nsfw_detection'
                                else self.model_dir)

    def _load(self):
        """加载 HuggingFace pipeline（线程安全，含熔断保护）"""
        if self._pipeline is not None:
            return
        if self._load_failed:
            raise RuntimeError("Falconsai 模型加载曾失败，已熔断，请检查模型文件后重启服务")

        with self._lock:
            if self._pipeline is not None:
                return
            try:
                logger.info("Falconsai: 开始加载模型 pipeline")
                from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
                model = AutoModelForImageClassification.from_pretrained(
                    self.model_dir, local_files_only=True
                )
                image_processor = AutoImageProcessor.from_pretrained(
                    self.model_dir, local_files_only=True
                )
                self._pipeline = pipeline(
                    "image-classification",
                    model=model,
                    image_processor=image_processor,
                    device="cpu",
                )
                logger.info("Falconsai: 模型 pipeline 加载完成")
            except Exception:
                self._load_failed = True
                logger.exception("Falconsai: 模型加载失败，已标记熔断")
                raise

    def detect(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        """
        二分类 NSFW 检测

        Returns:
            dict: 统一安全标签格式（二分类模型不返回性感字段）
        """
        if not os.path.exists(image_path):
            logger.warning("Falconsai: 图片文件不存在 %s", image_path)
            return {"status": "error", "message": "图片文件不存在"}

        t = thresholds if thresholds else self.thresholds

        try:
            start = time.time()
            self._load()
            file_size = os.path.getsize(image_path)

            # 图片加载
            with Image.open(image_path) as img:
                img_rgb = img.convert('RGB')

            try:
                # Pipeline 推理（Lock 串行化）
                with self._lock:
                    results = self._pipeline(img_rgb)
            finally:
                img_rgb.close()

            # 解析 pipeline 输出
            scores = {}
            for item in results:
                scores[item['label']] = round(float(item['score']), 4)

            nsfw_score = scores.get('nsfw', 0.0)
            normal_score = scores.get('normal', 0.0)

            # 阈值判定
            block_th = t.get('nsfw_block', 0.8)
            review_th = t.get('nsfw_review', 0.5)

            if nsfw_score >= block_th:
                action, action_text = 'block', '拦截'
                details = [f"NSFW {nsfw_score:.2%} >= 拦截阈值 {block_th:.2%}"]
            elif nsfw_score >= review_th:
                action, action_text = 'review', '复审'
                details = [f"NSFW {nsfw_score:.2%} >= 复审阈值 {review_th:.2%}"]
            else:
                action, action_text = 'pass', '放行'
                details = []

            elapsed = round(time.time() - start, 2)
            logger.info("Falconsai: action=%s, nsfw=%.4f, elapsed=%.2fs",
                        action, nsfw_score, elapsed)

            return {
                'status': 'success',
                'model': 'Falconsai ViT',
                'model_id': 'falconsai',
                'elapsed_seconds': elapsed,
                'image_size': file_size,
                'raw_scores': scores,
                'content_type': None,
                'safety': {
                    '色情': nsfw_score,
                    '正常': normal_score,
                },
                'action': action,
                'action_text': action_text,
                'details': details,
            }

        except Exception as e:
            logger.exception("Falconsai 检测失败")
            return {"status": "error", "message": f"Falconsai 检测失败: {str(e)}"}
