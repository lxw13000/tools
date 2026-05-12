"""
MobileNet V2 140 NSFW 检测模块

GantMan 5 分类模型（drawings / hentai / neutral / porn / sexy），
模型大小 ~17MB，输出 5 类概率，可派生「色情/性感/正常」安全分类和
「人物/动漫/风景」内容分类。

线程安全设计：
  - 模型加载使用 Lock 保护，防止多线程重复初始化
  - 模型推理使用 Lock 串行化，TF/Keras predict 非线程安全
  - 加载失败后标记熔断，避免反复重试消耗资源
"""

import os
import time
import logging
import threading
from typing import Dict, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 模型输出类别名称（与模型输出顺序一一对应）
GANTMAN_CLASSES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

# 模型输入尺寸
INPUT_SIZE = 224


class MobileNetDetector:
    """MobileNet V2 140 五分类 NSFW 检测器"""

    def __init__(self, weights_path: str, config: Dict = None):
        self.weights_path = weights_path
        self._model = None              # 模型懒加载
        self._load_failed = False       # 熔断标记
        self._load_lock = threading.Lock()   # 保护加载
        self._infer_lock = threading.Lock()  # 串行化推理

        # 默认阈值（与原 NSFWDetector.mobilenet_thresholds 对齐）
        self.thresholds = {
            'porn': 0.6,
            'hentai': 0.5,
            'sexy': 0.7,
            'porn_hentai': 0.65,
        }
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('thresholds', {})
            for key in self.thresholds:
                if key in t:
                    self.thresholds[key] = float(t[key])

        logger.info("MobileNet: 模型地址 = %s", self.weights_path)

    @staticmethod
    def check_files(weights_path: str) -> bool:
        """静态方法：检查权重文件是否存在（不触发实例化）"""
        return os.path.isfile(weights_path)

    def is_available(self) -> bool:
        """检查权重文件是否就绪"""
        return self.check_files(self.weights_path)

    def _ensure_loaded(self):
        """确保模型已加载（线程安全，含熔断保护）"""
        if self._model is not None:
            return
        if self._load_failed:
            raise RuntimeError("MobileNet 模型加载曾失败，已熔断，请检查权重文件后重启服务")

        with self._load_lock:
            if self._model is not None:
                return
            try:
                if not os.path.isfile(self.weights_path):
                    raise FileNotFoundError(
                        f"MobileNet 权重文件不存在: {self.weights_path}"
                    )
                logger.info("MobileNet: 开始加载模型")
                load_start = time.perf_counter()
                # tensorflow / hub 体积大，延迟到加载时导入
                import tensorflow as tf
                import tensorflow_hub as hub
                self._model = tf.keras.models.load_model(
                    self.weights_path,
                    custom_objects={'KerasLayer': hub.KerasLayer},
                    compile=False,
                )
                load_ms = int(round((time.perf_counter() - load_start) * 1000))
                logger.info("MobileNet: 模型加载完成，耗时 %dms", load_ms)
            except Exception:
                self._load_failed = True
                logger.exception("MobileNet: 模型加载失败，已标记熔断")
                raise

    def detect(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        """
        MobileNet V2 140 五分类检测

        Returns:
            dict: 统一格式检测结果（含 5 分类内容分类和 3 分类安全分类）
        """
        if not os.path.exists(image_path):
            logger.warning("MobileNet: 图片文件不存在 %s", image_path)
            return {"status": "error", "message": "图片文件不存在"}

        t = thresholds if thresholds else self.thresholds

        try:
            start = time.perf_counter()
            self._ensure_loaded()
            file_size = os.path.getsize(image_path)

            # 图片预处理：缩放到 224x224，归一化到 [-1, 1]
            with Image.open(image_path) as img:
                img_rgb = img.convert('RGB')
                img_resized = img_rgb.resize(
                    (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
                )
                arr = np.asarray(img_resized, dtype=np.float32)
                img_resized.close()
                img_rgb.close()

            arr = (arr / 127.5) - 1.0
            arr = np.expand_dims(arr, 0)

            # 模型推理（Lock 串行化，Keras predict 非线程安全）
            with self._infer_lock:
                preds = self._model.predict(arr, verbose=0)[0]

            raw = {name: round(float(preds[i]), 4)
                   for i, name in enumerate(GANTMAN_CLASSES)}
            elapsed_seconds_raw = time.perf_counter() - start
            elapsed = round(elapsed_seconds_raw, 2)
            elapsed_ms = int(round(elapsed_seconds_raw * 1000))

            # 内容分类映射
            content_type = {
                '人物': round(raw['porn'] + raw['sexy'] + raw['neutral'], 4),
                '动漫': round(raw['hentai'], 4),
                '风景': round(raw['drawings'], 4),
            }

            # 安全分类映射
            safety = {
                '色情': round(raw['porn'] + raw['hentai'], 4),
                '性感': round(raw['sexy'], 4),
                '正常': round(raw['neutral'] + raw['drawings'], 4),
            }

            action, action_text, details = self._decision(raw, t)

            logger.info(
                "MobileNet: action=%s, 色情=%.4f, 性感=%.4f, elapsed=%dms (%.2fs)",
                action, safety['色情'], safety['性感'], elapsed_ms, elapsed,
            )

            return {
                'status': 'success',
                'model': 'MobileNet V2 140',
                'model_id': 'mobilenet',
                'elapsed_seconds': elapsed,
                'elapsed_ms': elapsed_ms,
                'image_size': file_size,
                'raw_scores': raw,
                'content_type': content_type,
                'safety': safety,
                'action': action,
                'action_text': action_text,
                'details': details,
            }

        except Exception as e:
            logger.exception("MobileNet 检测失败")
            return {"status": "error", "message": f"MobileNet 检测失败: {str(e)}"}

    @staticmethod
    def _decision(raw: Dict, t: Dict):
        """
        5-class 阈值级联决策（优先级从高到低，命中即停止）

        Returns:
            (action, action_text, details) 三元组
        """
        details = []
        porn = raw.get('porn', 0)
        hentai = raw.get('hentai', 0)
        sexy = raw.get('sexy', 0)
        combined = porn + hentai

        pt = t.get('porn', 0.6)
        ht = t.get('hentai', 0.5)
        st = t.get('sexy', 0.7)
        ct = t.get('porn_hentai', 0.65)

        if porn > pt:
            action = 'block'
            details.append(f"色情 {porn:.2%} > 阈值 {pt:.2%}")
        elif combined > ct:
            action = 'block'
            details.append(f"色情+动漫 {combined:.2%} > 阈值 {ct:.2%}")
        elif hentai > ht:
            action = 'review'
            details.append(f"动漫 {hentai:.2%} > 阈值 {ht:.2%}")
        elif sexy > st:
            action = 'review'
            details.append(f"性感 {sexy:.2%} > 阈值 {st:.2%}")
        else:
            action = 'pass'

        action_text = {'block': '拦截', 'review': '复审', 'pass': '放行'}
        return action, action_text[action], details
