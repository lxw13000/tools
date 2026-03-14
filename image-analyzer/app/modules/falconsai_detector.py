"""
Falconsai ViT NSFW 检测模块
Falconsai/nsfw_image_detection, ViT-base, ~330MB, 二分类 nsfw/normal
"""

import os
import time
from typing import Dict, Optional
from PIL import Image


class FalconsaiDetector:

    def __init__(self, model_dir: str = '/app/models/falconsai', config: Dict = None):
        # 探测实际模型位置（snapshot_download 可能存为子目录）
        for sub in ['', 'nsfw_detection']:
            candidate = os.path.join(model_dir, sub) if sub else model_dir
            if os.path.exists(os.path.join(candidate, 'config.json')):
                self.model_dir = candidate
                break
        else:
            self.model_dir = model_dir

        self._pipeline = None
        self.thresholds = {
            'nsfw_block': 0.8,
            'nsfw_review': 0.5,
        }
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('falconsai', {}).get('thresholds', {})
            for key in ['nsfw_block', 'nsfw_review']:
                if key in t:
                    self.thresholds[key] = float(t[key])

    def is_available(self) -> bool:
        config_path = os.path.join(self.model_dir, 'config.json')
        safetensors = os.path.join(self.model_dir, 'model.safetensors')
        bin_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        return os.path.exists(config_path) and (
            os.path.exists(safetensors) or os.path.exists(bin_path)
        )

    def _load(self):
        if self._pipeline is not None:
            return
        if not self.is_available():
            raise FileNotFoundError(f"Falconsai 模型不存在: {self.model_dir}")
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

    def detect(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        """
        二分类 NSFW 检测

        Args:
            image_path: 图片文件绝对路径
            thresholds:  {nsfw_block, nsfw_review}

        Returns:
            dict: 统一安全标签格式，性感固定 0.0（二分类模型无法区分性感与色情）
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": "图片文件不存在"}

        t = thresholds if thresholds else self.thresholds

        try:
            start = time.time()
            self._load()
            file_size = os.path.getsize(image_path)
            img = Image.open(image_path).convert('RGB')
            results = self._pipeline(img)

            scores = {}
            for item in results:
                scores[item['label']] = round(float(item['score']), 4)

            nsfw_score = scores.get('nsfw', 0.0)
            normal_score = scores.get('normal', 0.0)

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
                    '性感': 0.0,  # 二分类模型无法区分性感与色情，固定 0.0
                    '暴力': 0.0,
                    '正常': normal_score,
                },
                'action': action,
                'action_text': action_text,
                'details': details,
            }

        except Exception as e:
            return {"status": "error", "message": f"Falconsai 检测失败: {str(e)}"}
