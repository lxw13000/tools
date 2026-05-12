"""鉴黄检测业务包

包含模型门面、各模型检测器、融合检测和对外服务。新增模型时在本包内新增
<name>_detector.py 并在 NSFWDetector 注册表登记即可。
"""

from .nsfw_detector import NSFWDetector
from .opennsfw2_detector import OpenNSFW2Detector
from .falconsai_detector import FalconsaiDetector
from .mobilenet_detector import MobileNetDetector
from .fusion_detector import FusionDetector
from .nsfw_service import NsfwService

__all__ = [
    'NSFWDetector',
    'OpenNSFW2Detector',
    'FalconsaiDetector',
    'MobileNetDetector',
    'FusionDetector',
    'NsfwService',
]
