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

设计要点：
  - 所有模型均为懒加载（首次调用时初始化），避免启动时占用过多内存
  - 阈值支持配置文件默认值 + API 调用时动态覆盖
  - 性感(sexy)独立参与阈值评判
"""

import os
import time
import logging
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
        'type': 'opennsfw2',           # 模型类型标识，路由用
        'accuracy': '较高',
        'speed': '快',
        'size': '~23MB',
        'desc': 'Yahoo OpenNSFW ResNet-50，二分类，自动下载',
        'output': 'binary',            # 输出类型：二分类（nsfw/normal）
    },
    'mobilenet': {
        'name': 'MobileNet V2 140',
        'type': 'tf_gantman',          # GantMan 的 TF 模型
        'file': 'mobilenet_v2_140_224.h5',
        'input_size': 224,             # 模型输入尺寸 224x224
        'accuracy': '较高',
        'speed': '最快',
        'size': '~17MB',
        'desc': 'GantMan 5 分类，可输出内容分类(人物/动漫/风景)',
        'output': '5class',            # 输出类型：5 分类
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
        """
        初始化 NSFW 检测门面

        Args:
            models_dir: 模型文件所在目录路径
            config:     全局配置字典（来自 config.yaml）
        """
        self.models_dir = models_dir
        self.config = config or {}

        # 三个模型实例均为懒加载（None 表示尚未初始化）
        self._tf_model = None       # MobileNet TF 模型对象
        self._opennsfw2 = None      # OpenNSFW2Detector 实例
        self._falconsai = None      # FalconsaiDetector 实例

        # MobileNet 5-class 阈值默认值（可通过配置文件覆盖）
        # porn:        色情单项 > 此值 → 拦截
        # hentai:      动漫色情 > 此值 → 复审
        # sexy:        性感 > 此值 → 复审
        # porn_hentai: 色情+动漫组合 > 此值 → 拦截
        self.mobilenet_thresholds = {
            'porn': 0.6,
            'hentai': 0.5,
            'sexy': 0.7,
            'porn_hentai': 0.65,
        }
        # 从配置文件读取阈值覆盖默认值
        if config and 'nsfw_detection' in config:
            t = config['nsfw_detection'].get('thresholds', {})
            for key in self.mobilenet_thresholds:
                if key in t:
                    self.mobilenet_thresholds[key] = float(t[key])

    # ---- 懒加载：首次使用时才初始化模型，避免启动时内存占用过高 ----

    def _get_opennsfw2(self):
        """获取 OpenNSFW2 检测器实例（懒加载）"""
        if self._opennsfw2 is None:
            from .opennsfw2_detector import OpenNSFW2Detector
            self._opennsfw2 = OpenNSFW2Detector(config=self.config)
        return self._opennsfw2

    def _get_falconsai(self):
        """获取 Falconsai ViT 检测器实例（懒加载）"""
        if self._falconsai is None:
            from .falconsai_detector import FalconsaiDetector
            self._falconsai = FalconsaiDetector(
                model_dir=os.path.join(self.models_dir, 'falconsai'),
                config=self.config,
            )
        return self._falconsai

    def _load_tf_model(self):
        """加载 MobileNet V2 140 TF 模型（懒加载）"""
        if self._tf_model is not None:
            return self._tf_model
        cfg = MODEL_REGISTRY['mobilenet']
        path = os.path.join(self.models_dir, cfg['file'])
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {cfg['file']}")
        # 加载 .h5 模型，需注册 KerasLayer 自定义对象
        self._tf_model = tf.keras.models.load_model(
            path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False
        )
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
        """检查指定模型是否可用（依赖已安装且模型文件存在）"""
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
            try:
                return self._get_falconsai().is_available()
            except Exception:
                return False
        elif cfg['type'] == 'tf_gantman':
            return os.path.exists(os.path.join(self.models_dir, cfg['file']))
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
            thresholds: 阈值覆盖字典（可选），key 含义取决于模型类型：
                        mobilenet:  {porn, hentai, sexy, porn_hentai}
                        二分类模型: {nsfw_block, nsfw_review}

        Returns:
            dict: 统一格式
                成功: {status:'success', model, model_id, elapsed_seconds, image_size,
                       raw_scores, content_type, safety{色情,性感,正常},
                       action, action_text, details}
                失败: {status:'error', message}
        """
        if model_id not in MODEL_REGISTRY:
            return {"status": "error", "message": f"未知模型: {model_id}"}

        cfg = MODEL_REGISTRY[model_id]

        # 根据模型类型路由到对应的检测实现
        if cfg['type'] == 'opennsfw2':
            return self._get_opennsfw2().detect(image_path, thresholds=thresholds)

        if cfg['type'] == 'falconsai':
            return self._get_falconsai().detect(image_path, thresholds=thresholds)

        if cfg['type'] == 'tf_gantman':
            return self._detect_mobilenet(image_path, thresholds)

        return {"status": "error", "message": f"未知模型类型: {cfg['type']}"}

    # ---- MobileNet V2 140 (5-class) 检测实现 ----

    def _detect_mobilenet(self, image_path: str, thresholds: Optional[Dict] = None) -> Dict:
        """
        MobileNet V2 140 五分类检测

        模型输出 5 个类别概率：drawings, hentai, neutral, porn, sexy
        映射为统一的安全分类和内容分类标签。

        Args:
            image_path: 图片文件绝对路径
            thresholds: 阈值覆盖字典 {porn, hentai, sexy, porn_hentai}
        """
        if not os.path.exists(image_path):
            return {"status": "error", "message": "图片文件不存在"}

        # 使用传入阈值，若未传则用配置文件/默认阈值
        t = thresholds if thresholds else self.mobilenet_thresholds

        try:
            start = time.time()
            model = self._load_tf_model()
            file_size = os.path.getsize(image_path)

            # ---- 图片预处理：缩放到 224x224，归一化到 [-1, 1] ----
            with Image.open(image_path) as img:
                img_rgb = img.convert('RGB')
                dim = MODEL_REGISTRY['mobilenet']['input_size']
                img_resized = img_rgb.resize((dim, dim), Image.Resampling.BILINEAR)
                arr = np.asarray(img_resized, dtype=np.float32)
            # MobileNet V2 预处理：像素值从 [0, 255] 归一化到 [-1, 1]
            arr = (arr / 127.5) - 1.0
            arr = np.expand_dims(arr, 0)  # 增加 batch 维度

            # ---- 模型推理 ----
            preds = model.predict(arr, verbose=0)[0]
            # 构建原始分数字典：{drawings: 0.xx, hentai: 0.xx, neutral: 0.xx, porn: 0.xx, sexy: 0.xx}
            raw = {name: round(float(preds[i]), 4) for i, name in enumerate(GANTMAN_CLASSES)}

            elapsed = round(time.time() - start, 2)

            # ---- 内容分类映射 ----
            # 人物 = porn + sexy + neutral（包含人物主体的类别）
            # 动漫 = hentai（动漫/插画类）
            # 风景 = drawings（风景/绘画类）
            content_type = {
                '人物': round(raw['porn'] + raw['sexy'] + raw['neutral'], 4),
                '动漫': round(raw['hentai'], 4),
                '风景': round(raw['drawings'], 4),
            }

            # ---- 安全分类映射 ----
            # 色情 = porn + hentai（包含真人色情和动漫色情）
            # 性感 = sexy（独立分出，参与阈值评判）
            # 正常 = neutral + drawings（安全内容）
            safety = {
                '色情': round(raw['porn'] + raw['hentai'], 4),
                '性感': round(raw['sexy'], 4),
                '正常': round(raw['neutral'] + raw['drawings'], 4),
            }

            # ---- 阈值级联决策 ----
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
            logger.exception("MobileNet 检测失败")
            return {"status": "error", "message": f"MobileNet 检测失败: {str(e)}"}

    def _mobilenet_decision(self, raw: Dict, t: Dict):
        """
        MobileNet 5-class 阈值级联决策

        按优先级从高到低依次判定（命中即停止）：
          1. porn 单项 > porn 阈值        → block（拦截）
          2. porn + hentai > 组合阈值      → block（拦截）
          3. hentai 单项 > hentai 阈值     → review（复审）
          4. sexy 单项 > sexy 阈值         → review（复审）
          5. 均未触发                       → pass（放行）

        Args:
            raw: 模型原始 5 分类分数 {drawings, hentai, neutral, porn, sexy}
            t:   阈值字典 {porn, hentai, sexy, porn_hentai}

        Returns:
            (action, action_text, details) 三元组
        """
        details = []

        porn = raw.get('porn', 0)
        hentai = raw.get('hentai', 0)
        sexy = raw.get('sexy', 0)
        combined = porn + hentai  # 色情 + 动漫色情组合分数

        # 读取阈值（支持部分覆盖，未覆盖的用默认值）
        pt = t.get('porn', 0.6)
        ht = t.get('hentai', 0.5)
        st = t.get('sexy', 0.7)
        ct = t.get('porn_hentai', 0.65)

        # 级联判定：优先级 porn > porn+hentai > hentai > sexy
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
