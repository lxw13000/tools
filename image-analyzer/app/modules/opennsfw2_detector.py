"""
OpenNSFW2 检测模块 (Yahoo OpenNSFW 的 TF2 port)
ResNet-50-thin, ~23MB, 二分类 NSFW 概率
"""

import os
import time
import numpy as np
from typing import Dict, Optional


class OpenNSFW2Detector:

    def __init__(self, config: Dict = None):
        self._model = None
        self.thresholds = {
            'nsfw_block': 0.8,
            'nsfw_review': 0.5,
        }
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('opennsfw2', {}).get('thresholds', {})
            for key in ['nsfw_block', 'nsfw_review']:
                if key in t:
                    self.thresholds[key] = float(t[key])

    def is_available(self) -> bool:
        try:
            import opennsfw2
            return True
        except ImportError:
            return False

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import opennsfw2 as n2
        self._model = n2.make_open_nsfw_model()

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
            self._ensure_loaded()

            import opennsfw2 as n2
            from PIL import Image
            pil_image = Image.open(image_path).convert('RGB')
            image = n2.preprocess_image(pil_image, n2.Preprocessing.YAHOO)
            inputs = np.expand_dims(image, axis=0)
            predictions = self._model.predict(inputs, verbose=0)
            nsfw_prob = round(float(predictions[0][1]), 4)
            normal_prob = round(1.0 - nsfw_prob, 4)

            file_size = os.path.getsize(image_path)
            elapsed = round(time.time() - start, 2)

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

            return {
                'status': 'success',
                'model': 'OpenNSFW2 (Yahoo)',
                'model_id': 'opennsfw2',
                'elapsed_seconds': elapsed,
                'image_size': file_size,
                'raw_scores': {'nsfw': nsfw_prob, 'normal': normal_prob},
                'content_type': None,
                'safety': {
                    '色情': nsfw_prob,
                    '性感': 0.0,  # 二分类模型无法区分性感与色情，固定 0.0
                    '暴力': 0.0,
                    '正常': normal_prob,
                },
                'action': action,
                'action_text': action_text,
                'details': details,
            }

        except Exception as e:
            return {"status": "error", "message": f"OpenNSFW2 检测失败: {str(e)}"}
