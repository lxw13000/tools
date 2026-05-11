"""
NSFW 检测服务门面 (Facade)

统一管理 3 个检测模型：OpenNSFW2、MobileNet V2 140、Falconsai ViT，
对外提供统一的 detect() 接口，内部路由到对应模型实现。

输出统一的双分类标签：
  - 安全分类：色情 / 性感 / 正常（用于安全策略判定）
  - 内容分类：人物 / 动漫 / 风景（仅 MobileNet 5-class 模型支持）

判定结果（action）：
  - 'block'  — 拦截（超过拦截阈值）
  - 'review' — 人工复核（超过复审阈值但未达拦截）
  - 'pass'   — 放行（所有指标均低于复审阈值）
  - status='error' 时表示检测失败（模型加载/推理异常等）

线程安全设计：
  - 模型懒加载使用 Lock 保护，防止多线程重复初始化
  - 模型推理使用 Lock 串行化，因 TF/Keras predict() 非线程安全
"""

import os
import time
import logging
import threading
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger(__name__)

# ---- 模型注册表：描述每个模型的元信息，用于模型管理和前端展示 ----
MODEL_REGISTRY = {
    'opennsfw2': {
        'name': 'OpenNSFW2 (Yahoo)',
        'type': 'opennsfw2',
        'file': 'open_nsfw_weights.h5',
        'accuracy': '较高',
        'speed': '快',
        'size': '~23MB',
        'desc': 'Yahoo OpenNSFW ResNet-50，二分类',
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

# MobileNet 5-class 输出的类别名称（与模型输出顺序对应）
GANTMAN_CLASSES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']


class NSFWDetector:
    """NSFW 检测门面：管理所有检测器实例，路由 detect() 调用，输出统一标签格式"""

    def __init__(self, models_dir: str = '/app/models', config: Dict = None):
        self.models_dir = models_dir
        self.config = config or {}

        # 模型实例（懒加载，None 表示尚未初始化）
        self._tf_model = None
        self._opennsfw2 = None
        self._falconsai = None

        # 线程锁：保护懒加载和模型推理的线程安全
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

        # MobileNet 5-class 默认阈值
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

        logger.info("NSFWDetector 初始化完成, models_dir=%s", models_dir)

    # ---- 懒加载（Lock 保护，防止多线程重复初始化）----

    def _get_opennsfw2(self):
        """获取 OpenNSFW2 检测器实例（线程安全懒加载）"""
        if self._opennsfw2 is not None:
            return self._opennsfw2
        with self._load_lock:
            if self._opennsfw2 is None:
                logger.info("OpenNSFW2: 首次调用，初始化检测器")
                from .opennsfw2_detector import OpenNSFW2Detector
                weights_path = os.path.join(
                    self.models_dir, MODEL_REGISTRY['opennsfw2']['file']
                )
                self._opennsfw2 = OpenNSFW2Detector(
                    weights_path=weights_path, config=self.config,
                )
        return self._opennsfw2

    def _get_falconsai(self):
        """获取 Falconsai ViT 检测器实例（线程安全懒加载）"""
        if self._falconsai is not None:
            return self._falconsai
        with self._load_lock:
            if self._falconsai is None:
                logger.info("Falconsai: 首次调用，初始化检测器")
                from .falconsai_detector import FalconsaiDetector
                self._falconsai = FalconsaiDetector(
                    model_dir=os.path.join(self.models_dir, 'falconsai'),
                    config=self.config,
                )
        return self._falconsai

    def _load_tf_model(self):
        """加载 MobileNet V2 140 TF 模型（线程安全懒加载）"""
        if self._tf_model is not None:
            return self._tf_model
        with self._load_lock:
            if self._tf_model is None:
                cfg = MODEL_REGISTRY['mobilenet']
                path = os.path.join(self.models_dir, cfg['file'])
                if not os.path.exists(path):
                    raise FileNotFoundError(f"模型文件不存在: {cfg['file']}")
                logger.info("MobileNet: 开始加载模型 %s", cfg['file'])
                self._tf_model = tf.keras.models.load_model(
                    path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False
                )
                logger.info("MobileNet: 模型加载完成")
        return self._tf_model

    # ---- 公开 API ----

    def get_models_info(self) -> List[Dict]:
        """返回所有已注册模型的元信息列表（含运行时可用性检查）"""
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
        """检查指定模型是否可用（仅检查依赖和文件，不触发模型加载）"""
        cfg = MODEL_REGISTRY.get(model_id)
        if not cfg:
            return False
        try:
            if cfg['type'] == 'opennsfw2':
                try:
                    import opennsfw2  # noqa: F401
                except ImportError:
                    return False
                from .opennsfw2_detector import OpenNSFW2Detector
                return OpenNSFW2Detector.check_files(
                    os.path.join(self.models_dir, cfg['file'])
                )
            elif cfg['type'] == 'falconsai':
                # 仅检查文件存在性，不触发完整初始化
                from .falconsai_detector import FalconsaiDetector
                return FalconsaiDetector.check_files(
                    os.path.join(self.models_dir, 'falconsai')
                )
            elif cfg['type'] == 'tf_gantman':
                return os.path.exists(os.path.join(self.models_dir, cfg['file']))
        except Exception:
            return False
        return False

    def get_default_thresholds(self) -> Dict:
        """返回 MobileNet 5-class 当前阈值配置（供前端渲染默认滑块值）"""
        return dict(self.mobilenet_thresholds)

    def detect(self, image_path: str, model_id: str = 'mobilenet',
               thresholds: Optional[Dict] = None) -> Dict:
        """
        统一检测入口，根据 model_id 路由到对应模型

        Args:
            image_path: 图片文件绝对路径
            model_id:   模型标识 'mobilenet' | 'opennsfw2' | 'falconsai'
            thresholds: 阈值覆盖字典（可选）

        Returns:
            dict: 统一格式的检测结果
        """
        if model_id not in MODEL_REGISTRY:
            logger.warning("NSFWDetector: 未知模型 %s", model_id)
            return {"status": "error", "message": f"未知模型: {model_id}"}

        cfg = MODEL_REGISTRY[model_id]
        logger.debug("NSFWDetector: 开始检测, model=%s, image=%s", model_id, image_path)

        # 路由到对应模型，统一捕获初始化异常
        try:
            if cfg['type'] == 'opennsfw2':
                return self._get_opennsfw2().detect(image_path, thresholds=thresholds)
            if cfg['type'] == 'falconsai':
                return self._get_falconsai().detect(image_path, thresholds=thresholds)
            if cfg['type'] == 'tf_gantman':
                return self._detect_mobilenet(image_path, thresholds)
        except Exception as e:
            logger.exception("NSFWDetector: 模型 %s 检测失败", model_id)
            return {"status": "error", "message": f"{cfg['name']} 检测失败: {str(e)}"}

        return {"status": "error", "message": f"未知模型类型: {cfg['type']}"}

    # ---- MobileNet V2 140 (5-class) 检测实现 ----

    def _detect_mobilenet(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        """MobileNet V2 140 五分类检测（含线程安全推理）"""
        if not os.path.exists(image_path):
            logger.warning("MobileNet: 图片文件不存在 %s", image_path)
            return {"status": "error", "message": "图片文件不存在"}

        t = thresholds if thresholds else self.mobilenet_thresholds

        try:
            start = time.time()
            model = self._load_tf_model()
            file_size = os.path.getsize(image_path)

            # 图片预处理：缩放到 224x224，归一化到 [-1, 1]
            with Image.open(image_path) as img:
                img_rgb = img.convert('RGB')
                dim = MODEL_REGISTRY['mobilenet']['input_size']
                img_resized = img_rgb.resize((dim, dim), Image.Resampling.BILINEAR)
                arr = np.asarray(img_resized, dtype=np.float32)
                img_resized.close()
                img_rgb.close()

            arr = (arr / 127.5) - 1.0
            arr = np.expand_dims(arr, 0)

            # 模型推理（Lock 串行化，Keras predict 非线程安全）
            with self._infer_lock:
                preds = model.predict(arr, verbose=0)[0]

            raw = {name: round(float(preds[i]), 4) for i, name in enumerate(GANTMAN_CLASSES)}
            elapsed = round(time.time() - start, 2)

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

            action, action_text, details = self._mobilenet_decision(raw, t)

            logger.info("MobileNet: action=%s, 色情=%.4f, 性感=%.4f, elapsed=%.2fs",
                        action, safety['色情'], safety['性感'], elapsed)

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
            logger.exception("MobileNet 检测失败")
            return {"status": "error", "message": f"MobileNet 检测失败: {str(e)}"}

    def _mobilenet_decision(self, raw: Dict, t: Dict):
        """
        MobileNet 5-class 阈值级联决策（优先级从高到低，命中即停止）

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
