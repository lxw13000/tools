"""modules 包统一入口

业务模块拆分：
    - nsfw   鉴黄检测（多模型 + 融合 + 对外服务）
    - motion 动静态检测（融合评分 + 人脸增强 + 定时任务）

为减小迁移影响，本文件继续按平铺方式重导出各类，调用方无需感知子包结构。
"""

from .nsfw import (
    NSFWDetector,
    OpenNSFW2Detector,
    FalconsaiDetector,
    MobileNetDetector,
    FusionDetector,
    NsfwService,
)
from .motion import (
    MotionDetector,
    FaceDetector,
    SchedulerService,
)

__all__ = [
    'MotionDetector',
    'FaceDetector',
    'NSFWDetector',
    'OpenNSFW2Detector',
    'FalconsaiDetector',
    'MobileNetDetector',
    'FusionDetector',
    'SchedulerService',
    'NsfwService',
]
