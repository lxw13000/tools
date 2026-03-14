"""
NSFW 检测服务门面 (Facade)
统一管理 3 个检测模型: OpenNSFW2, MobileNet V2 140, Falconsai ViT
提供双分类标签输出: 安全分类(色情/暴力/正常) + 内容分类(人物/动漫/风景)
"""

import os
import time
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
import tensorflow as tf
import tensorflow_hub as hub


# ---- 模型注册表 ----

MODEL_REGISTRY = {
    'opennsfw2': {
        'name': 'OpenNSFW2 (Yahoo)',
        'type': 'opennsfw2',
        'accuracy': '较高',
        'speed': '快',
        'size': '~23MB',
        'desc': 'Yahoo OpenNSFW ResNet-50，二分类，自动下载',
        'output': 'binary',
    },
    'mobilenet': {
        'name': 'MobileNet V2 140',
        'type': 'tf_gantman',
        'file': 'mobilenet_v2_140_224.h5',
        'input_size': 224,
        'accuracy': '较高',
        'speed': '最快',
        'size': '~17MB',
        'desc': 'GantMan 5 分类，可输出内容分类(人物/动漫/风景)',
        'output': '5class',
    },
    'falconsai': {
        'name': 'Falconsai ViT',
        'type': 'falconsai',
        'accuracy': '最高',
        'speed': '较慢',
        'size': '~330MB',
        'desc': 'Vision Transformer，98%+ 精度，二分类',
        'output': 'binary',
    },
}

GANTMAN_CLASSES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']


class NSFWDetector:
    """中央门面：管理所有检测器，路由 detect() 调用，输出统一标签格式"""

    def __init__(self, models_dir: str = '/app/models', config: Dict = None):
        self.models_dir = models_dir
        self.config = config or {}

        self._tf_model = None
        self._opennsfw2 = None
        self._falconsai = None

        # MobileNet 5-class 阈值
        self.mobilenet_thresholds = {
            'porn': 0.6,
            'hentai': 0.5,
            'sexy': 0.7,
            'porn_hentai': 0.65,
        }
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('thresholds', {})
            for key in self.mobilenet_thresholds:
                if key in t:
                    self.mobilenet_thresholds[key] = float(t[key])

    # ---- 延迟加载 ----

    def _get_opennsfw2(self):
        if self._opennsfw2 is None:
            from .opennsfw2_detector import OpenNSFW2Detector
            self._opennsfw2 = OpenNSFW2Detector(config=self.config)
        return self._opennsfw2

    def _get_falconsai(self):
        if self._falconsai is None:
            from .falconsai_detector import FalconsaiDetector
            self._falconsai = FalconsaiDetector(
                model_dir=os.path.join(self.models_dir, 'falconsai'),
                config=self.config,
            )
        return self._falconsai

    def _load_tf_model(self):
        if self._tf_model is not None:
            return self._tf_model
        cfg = MODEL_REGISTRY['mobilenet']
        path = os.path.join(self.models_dir, cfg['file'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {cfg['file']}")
        self._tf_model = tf.keras.models.load_model(
            path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False
        )
        return self._tf_model

    # ---- 公开 API ----

    def get_models_info(self) -> List[Dict]:
        result = []
        for model_id, cfg in MODEL_REGISTRY.items():
            result.append({
                'id': model_id,
                'name': cfg['name'],
                'accuracy': cfg['accuracy'],
                'speed': cfg['speed'],
                'size': cfg['size'],
                'desc': cfg['desc'],
                'output': cfg['output'],
                'available': self._check_available(model_id),
            })
        return result

    def _check_available(self, model_id: str) -> bool:
        cfg = MODEL_REGISTRY.get(model_id)
        if not cfg:
            return False
        if cfg['type'] == 'opennsfw2':
            try:
                import opennsfw2
                return True
            except ImportError:
                return False
        elif cfg['type'] == 'falconsai':
            return self._get_falconsai().is_available()
        elif cfg['type'] == 'tf_gantman':
            return os.path.exists(os.path.join(self.models_dir, cfg['file']))
        return False

    def get_default_thresholds(self) -> Dict:
        return dict(self.mobilenet_thresholds)

    def detect(self, image_path: str, model_id: str = 'mobilenet',
               thresholds: Optional[Dict] = None) -> Dict:
        if model_id not in MODEL_REGISTRY:
            return {"status": "error", "message": f"未知模型: {model_id}"}

        cfg = MODEL_REGISTRY[model_id]

        if cfg['type'] == 'opennsfw2':
            return self._get_opennsfw2().detect(image_path, thresholds=thresholds)

        if cfg['type'] == 'falconsai':
            return self._get_falconsai().detect(image_path, thresholds=thresholds)

        if cfg['type'] == 'tf_gantman':
            return self._detect_mobilenet(image_path, thresholds)

        return {"status": "error", "message": f"未知模型类型: {cfg['type']}"}

    # ---- MobileNet V2 140 (5-class) ----

    def _detect_mobilenet(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        if not os.path.exists(image_path):
            return {"status": "error", "message": "图片文件不存在"}

        t = thresholds if thresholds else self.mobilenet_thresholds

        try:
            start = time.time()
            model = self._load_tf_model()
            file_size = os.path.getsize(image_path)

            img = Image.open(image_path).convert('RGB')
            dim = MODEL_REGISTRY['mobilenet']['input_size']
            img_resized = img.resize((dim, dim), Image.BILINEAR)
            arr = np.asarray(img_resized, dtype=np.float32)
            arr = (arr / 127.5) - 1.0
            arr = np.expand_dims(arr, 0)

            preds = model.predict(arr, verbose=0)[0]
            raw = {name: round(float(preds[i]), 4) for i, name in enumerate(GANTMAN_CLASSES)}

            elapsed = round(time.time() - start, 2)

            # 双分类标签映射
            content_type = {
                '人物': round(raw['porn'] + raw['sexy'] + raw['neutral'], 4),
                '动漫': round(raw['hentai'], 4),
                '风景': round(raw['drawings'], 4),
            }
            safety = {
                '色情': round(raw['porn'] + raw['hentai'] + raw['sexy'], 4),
                '暴力': 0.0,
                '正常': round(raw['neutral'] + raw['drawings'], 4),
            }

            action, action_text, details = self._mobilenet_decision(raw, t)

            return {
                'status': 'success',
                'model': 'MobileNet V2 140',
                'model_id': 'mobilenet',
                'elapsed_seconds': elapsed,
                'image_size': file_size,
                'raw_scores': raw,
                'content_type': content_type,
                'safety': safety,
                'action': action,
                'action_text': action_text,
                'details': details,
            }

        except Exception as e:
            return {"status": "error", "message": f"MobileNet 检测失败: {str(e)}"}

    def _mobilenet_decision(self, raw: Dict, t: Dict):
        action = 'pass'
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

        action_text = {'block': '拦截', 'review': '复审', 'pass': '放行'}
        return action, action_text[action], details
