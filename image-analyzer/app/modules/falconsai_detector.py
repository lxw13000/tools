"""
Falconsai ViT NSFW 检测模块

基于 Falconsai/nsfw_image_detection 模型，使用 Vision Transformer (ViT-base) 架构，
模型大小 ~330MB，精度 98%+，输出二分类 nsfw / normal。

二分类模型特性：
  - 只输出 nsfw / normal 两个概率值
  - 无法区分「色情」和「性感」，故安全分类中不返回性感字段（避免 0.0 造成理解偏差）
  - 内容分类（人物/动漫/风景）不支持，返回 None

模型加载：
  - 使用 HuggingFace transformers pipeline（本地离线加载）
  - 支持 model.safetensors 和 pytorch_model.bin 两种格式
  - 自动探测 snapshot_download 可能存储的子目录结构

阈值判定：
  - nsfw_score >= nsfw_block  → 拦截
  - nsfw_score >= nsfw_review → 复审
  - 其余                       → 放行
"""

import os
import time
import logging
from typing import Dict, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class FalconsaiDetector:
    """Falconsai ViT 二分类 NSFW 检测器"""

    def __init__(self, model_dir: str = '/app/models/falconsai', config: Dict = None):
        """
        初始化 Falconsai 检测器

        Args:
            model_dir: 模型文件目录路径
            config:    全局配置字典，从中读取 nsfw_detection.falconsai.thresholds
        """
        # 探测实际模型位置（snapshot_download 可能将模型存在子目录中）
        for sub in ['', 'nsfw_detection']:
            candidate = os.path.join(model_dir, sub) if sub else model_dir
            if os.path.exists(os.path.join(candidate, 'config.json')):
                self.model_dir = candidate
                break
        else:
            self.model_dir = model_dir

        self._pipeline = None  # HuggingFace pipeline 懒加载

        # 默认阈值
        self.thresholds = {
            'nsfw_block': 0.8,   # NSFW >= 此值 → 拦截
            'nsfw_review': 0.5,  # NSFW >= 此值 → 复审
        }
        # 从配置文件覆盖默认阈值
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('falconsai', {}).get('thresholds', {})
            for key in ['nsfw_block', 'nsfw_review']:
                if key in t:
                    self.thresholds[key] = float(t[key])

    def is_available(self) -> bool:
        """检查模型文件是否存在（需要 config.json + 权重文件）"""
        config_path = os.path.join(self.model_dir, 'config.json')
        safetensors = os.path.join(self.model_dir, 'model.safetensors')
        bin_path = os.path.join(self.model_dir, 'pytorch_model.bin')
        return os.path.exists(config_path) and (
            os.path.exists(safetensors) or os.path.exists(bin_path)
        )

    def _load(self):
        """加载 HuggingFace image-classification pipeline（懒加载，仅首次调用时执行）"""
        if self._pipeline is not None:
            return
        if not self.is_available():
            raise FileNotFoundError(f"Falconsai 模型不存在: {self.model_dir}")
        from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
        # 加载模型和图像处理器（离线模式，不访问网络）
        model = AutoModelForImageClassification.from_pretrained(
            self.model_dir, local_files_only=True
        )
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_dir, local_files_only=True
        )
        # 构建分类 pipeline，强制使用 CPU
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
            thresholds: 阈值覆盖字典 {nsfw_block, nsfw_review}

        Returns:
            dict: 统一安全标签格式（二分类模型不返回性感字段）
                成功: {status:'success', model, model_id, elapsed_seconds, image_size,
                       raw_scores, content_type(None), safety{色情,正常},
                       action, action_text, details}
                失败: {status:'error', message}
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": "图片文件不存在"}

        # 使用传入阈值，若未传则用配置/默认阈值
        t = thresholds if thresholds else self.thresholds

        try:
            start = time.time()
            self._load()
            file_size = os.path.getsize(image_path)

            # ---- 图片加载并推理 ----
            with Image.open(image_path) as img:
                img_rgb = img.convert('RGB')
                # pipeline 接受 PIL Image 对象，内部完成预处理
                results = self._pipeline(img_rgb)

            # 解析 pipeline 输出 [{label: 'nsfw', score: 0.xx}, {label: 'normal', score: 0.xx}]
            scores = {}
            for item in results:
                scores[item['label']] = round(float(item['score']), 4)

            nsfw_score = scores.get('nsfw', 0.0)
            normal_score = scores.get('normal', 0.0)

            # ---- 阈值判定 ----
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
                'content_type': None,  # 二分类模型不支持内容分类
                # 二分类模型仅输出色情/正常，无法区分性感，故不返回性感字段（避免 0.0 造成理解偏差）
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
