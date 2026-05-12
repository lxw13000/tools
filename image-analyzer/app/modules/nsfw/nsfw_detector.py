"""
NSFW 检测服务门面 (Facade)

统一管理多个 NSFW 检测模型，对外提供统一的 detect() 接口，内部路由到
对应 detector 子类实现。当前注册的模型：
  - OpenNSFW2     ResNet-50，二分类
  - MobileNet V2  GantMan 5 分类
  - Falconsai ViT 二分类

新增模型只需：
  1. 在 modules/nsfw/ 下新增 <name>_detector.py，类暴露
     __init__(weights/model_dir, config) / is_available() / check_files() /
     detect(image_path, thresholds=None) 接口
  2. 在 MODEL_REGISTRY 注册元信息
  3. 在本类新增对应的 _get_<name>() 懒加载方法和 detect() 路由分支
"""

import os
import logging
import threading
from typing import Dict, Optional, List

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
        'type': 'mobilenet',
        'file': 'mobilenet_v2_140_224.h5',
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


def _read_mobilenet_default_thresholds(config: Dict) -> Dict:
    """从 config 解析 MobileNet 默认阈值（不实例化模型即可读取，供 API 使用）"""
    thresholds = {
        'porn': 0.6,
        'hentai': 0.5,
        'sexy': 0.7,
        'porn_hentai': 0.65,
    }
    if config and 'nsfw_detection' in config:
        t = config['nsfw_detection'].get('thresholds', {})
        for key in thresholds:
            if key in t:
                thresholds[key] = float(t[key])
    return thresholds


def _read_binary_default_thresholds(config: Dict, model_key: str) -> Dict:
    """从 config 解析二分类模型默认阈值（opennsfw2 / falconsai）"""
    thresholds = {'nsfw_block': 0.8, 'nsfw_review': 0.5}
    if config and 'nsfw_detection' in config:
        t = config['nsfw_detection'].get(model_key, {}).get('thresholds', {})
        for key in thresholds:
            if key in t:
                thresholds[key] = float(t[key])
    return thresholds


def _read_fusion_defaults(config: Dict) -> Dict:
    """从 config 解析融合默认权重 / 阈值 / 策略"""
    weights = {'opennsfw2': 0.25, 'mobilenet': 0.30, 'falconsai': 0.45}
    thresholds = {'block': 0.7, 'review': 0.4}
    strategy = 'weighted_average'
    if config and 'nsfw_detection' in config:
        fusion = config['nsfw_detection'].get('fusion', {})
        for k, v in (fusion.get('weights') or {}).items():
            if k in weights:
                weights[k] = float(v)
        for k, v in (fusion.get('thresholds') or {}).items():
            if k in thresholds:
                thresholds[k] = float(v)
        if 'strategy' in fusion:
            strategy = fusion['strategy']
    return {'weights': weights, 'thresholds': thresholds, 'strategy': strategy}


class NSFWDetector:
    """NSFW 检测门面：路由 detect() 调用到具体模型，输出统一标签格式"""

    def __init__(self, models_dir: str = '/app/models', config: Dict = None):
        self.models_dir = models_dir
        self.config = config or {}

        # 各 detector 实例（懒加载，None 表示尚未初始化）
        self._mobilenet = None
        self._opennsfw2 = None
        self._falconsai = None

        # 保护懒加载的线程安全
        self._load_lock = threading.Lock()

        # 保留供 /api/nsfw/config 旧接口使用（前端默认滑块值）
        self.mobilenet_thresholds = _read_mobilenet_default_thresholds(self.config)

        logger.info("NSFWDetector 初始化完成, models_dir=%s", models_dir)

    # ---- 懒加载（Lock 保护，防止多线程重复初始化）----

    def _get_opennsfw2(self):
        """获取 OpenNSFW2 检测器实例（线程安全懒加载）"""
        if self._opennsfw2 is not None:
            return self._opennsfw2
        with self._load_lock:
            if self._opennsfw2 is None:
                logger.info("NSFWDetector: OpenNSFW2 首次调用，初始化检测器")
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
                logger.info("NSFWDetector: Falconsai 首次调用，初始化检测器")
                from .falconsai_detector import FalconsaiDetector
                self._falconsai = FalconsaiDetector(
                    model_dir=os.path.join(self.models_dir, 'falconsai'),
                    config=self.config,
                )
        return self._falconsai

    def _get_mobilenet(self):
        """获取 MobileNet 检测器实例（线程安全懒加载）"""
        if self._mobilenet is not None:
            return self._mobilenet
        with self._load_lock:
            if self._mobilenet is None:
                logger.info("NSFWDetector: MobileNet 首次调用，初始化检测器")
                from .mobilenet_detector import MobileNetDetector
                weights_path = os.path.join(
                    self.models_dir, MODEL_REGISTRY['mobilenet']['file']
                )
                self._mobilenet = MobileNetDetector(
                    weights_path=weights_path, config=self.config,
                )
        return self._mobilenet

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
                from .falconsai_detector import FalconsaiDetector
                return FalconsaiDetector.check_files(
                    os.path.join(self.models_dir, 'falconsai')
                )
            elif cfg['type'] == 'mobilenet':
                from .mobilenet_detector import MobileNetDetector
                return MobileNetDetector.check_files(
                    os.path.join(self.models_dir, cfg['file'])
                )
        except Exception:
            return False
        return False

    def get_default_thresholds(self) -> Dict:
        """返回 MobileNet 5-class 当前阈值（保留向后兼容，仅 MobileNet）"""
        return dict(self.mobilenet_thresholds)

    def get_all_default_thresholds(self) -> Dict:
        """
        返回所有模型的默认阈值 + 融合默认配置

        供 /api/nsfw/config 使用，让前端阈值滑块的初始值与 config.yaml 一致。
        """
        return {
            'mobilenet': _read_mobilenet_default_thresholds(self.config),
            'opennsfw2': _read_binary_default_thresholds(self.config, 'opennsfw2'),
            'falconsai': _read_binary_default_thresholds(self.config, 'falconsai'),
            'fusion': _read_fusion_defaults(self.config),
        }

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
            if cfg['type'] == 'mobilenet':
                return self._get_mobilenet().detect(image_path, thresholds=thresholds)
        except Exception as e:
            logger.exception("NSFWDetector: 模型 %s 检测失败", model_id)
            return {"status": "error", "message": f"{cfg['name']} 检测失败: {str(e)}"}

        return {"status": "error", "message": f"未知模型类型: {cfg['type']}"}
