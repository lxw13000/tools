"""
人脸变化检测模块

独立模块，用于检测图片序列中人脸的变化情况（数量变化、位置移动、大小改变）。
作为动态检测的可选增强手段：如果图片中人脸发生了显著变化，说明画面是动态的。

设计原则：
  - 独立部署：可单独导入使用，不依赖 MotionDetector
  - 可配置：通过 config 字典控制所有参数
  - 容错：mediapipe 未安装时优雅降级，返回空结果
  - CPU-only：使用 MediaPipe 轻量级模型，无需 GPU

返回结果：
  - face_change_score: 0.0 ~ 1.0（0 = 无变化，1 = 显著变化）
  - pair_details: 逐对比较的详细信息
"""

import logging
import math
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

logger = logging.getLogger(__name__)

# ---- Lazy import mediapipe（允许未安装时优雅降级）----
_mp = None
_mp_available = None


def _ensure_mediapipe() -> bool:
    """延迟导入 mediapipe，缓存结果。"""
    global _mp, _mp_available
    if _mp_available is not None:
        return _mp_available
    try:
        import mediapipe as mp
        _mp = mp
        _mp_available = True
        logger.info("MediaPipe 加载成功，人脸检测增强可用")
    except ImportError:
        _mp_available = False
        logger.warning("MediaPipe 未安装，人脸检测增强不可用。安装命令: pip install mediapipe")
    return _mp_available


class FaceDetector:
    """图片序列人脸变化检测器"""

    def __init__(self, config: dict = None):
        """
        初始化人脸变化检测器

        Args:
            config: face_detection 配置字典
                - enabled: 是否启用（默认 False）
                - max_adjustment: 最大评分调整幅度（默认 0.15）
                - min_confidence: 人脸检测置信度阈值（默认 0.5）
                - position_threshold: 人脸中心偏移阈值，占图片比例（默认 0.1）
                - size_threshold: 人脸面积变化阈值，相对比例（默认 0.2）
        """
        config = config or {}
        self.enabled = bool(config.get('enabled', False))
        self.max_adjustment = float(config.get('max_adjustment', 0.15))
        self.min_confidence = float(config.get('min_confidence', 0.5))
        self.position_threshold = float(config.get('position_threshold', 0.05))
        self.size_threshold = float(config.get('size_threshold', 0.15))
        # 合成挂播检测参数
        self.static_face_ssim = float(config.get('static_face_ssim', 0.98))
        self.static_face_adjustment = float(config.get('static_face_adjustment', 0.30))

    def is_available(self) -> bool:
        """检查人脸检测是否已启用且 mediapipe 已安装。"""
        return self.enabled and _ensure_mediapipe()

    def detect_face_changes(self, image_paths: List[str]) -> Dict:
        """
        检测图片序列中人脸的变化情况

        对每张图片进行人脸检测，逐对比较相邻帧的人脸数量、位置、大小变化，
        综合评估画面是否存在真实的人脸运动。

        Args:
            image_paths: 图片文件路径列表（至少 2 张）

        Returns:
            dict:
                - face_change_score: float (0.0 ~ 1.0)
                - pair_details: 逐对详情列表
                - face_counts: 每张图片的人脸数量
                - available: 是否可用
        """
        if not _ensure_mediapipe():
            return {"face_change_score": 0.0, "pair_details": [], "face_counts": [], "available": False}

        if len(image_paths) < 2:
            return {"face_change_score": 0.0, "pair_details": [], "face_counts": [], "available": True}

        # 对每张图片检测人脸
        all_faces = []
        face_counts = []
        for path in image_paths:
            faces = self._detect_faces_in_image(path)
            all_faces.append(faces)
            face_counts.append(len(faces))

        # 逐对比较
        pair_details = []
        pair_scores = []
        for i in range(len(all_faces) - 1):
            detail = self._compare_pair(all_faces[i], all_faces[i + 1], i)
            pair_details.append(detail)
            pair_scores.append(detail['pair_score'])

        # 取最大值作为最终分数（任一对有变化即视为动态）
        face_change_score = max(pair_scores) if pair_scores else 0.0

        return {
            "face_change_score": round(face_change_score, 4),
            "pair_details": pair_details,
            "face_counts": face_counts,
            "available": True,
        }

    def compute_adjustment(self, face_change_score: float) -> float:
        """
        将人脸变化分数映射为融合评分的调整量

        Args:
            face_change_score: 人脸变化分数 (0.0 ~ 1.0)

        Returns:
            评分调整量（正数，用于从融合评分中减去）
        """
        return face_change_score * self.max_adjustment

    def detect_static_faces(self, image_paths: List[str]) -> Dict:
        """
        合成挂播检测：检测人脸区域是否在帧间像素级冻结

        场景：静态人像 + 动态特效叠加。画面整体看似"动态"，
        但人脸区域的像素内容在帧间完全不变。

        方法：裁剪每帧的人脸区域，逐对计算 SSIM。
        若所有帧对的人脸 SSIM 均超阈值 → 判定为合成挂播。

        Args:
            image_paths: 图片文件路径列表（至少 2 张）

        Returns:
            dict:
                - face_static_score: float (0.0 ~ 1.0)，越高越冻结
                - pair_details: 逐对人脸区域 SSIM
                - face_counts: 每张图片的人脸数量
                - has_static_face: bool
        """
        empty = {"face_static_score": 0.0, "pair_details": [], "face_counts": [],
                 "has_static_face": False}

        if not _ensure_mediapipe():
            return {**empty, "available": False}

        if len(image_paths) < 2:
            return empty

        # 检测每帧人脸
        all_faces = []
        face_counts = []
        for path in image_paths:
            faces = self._detect_faces_in_image(path)
            all_faces.append(faces)
            face_counts.append(len(faces))

        # 任何一帧没有人脸 → 无法做合成判定
        if any(c == 0 for c in face_counts):
            return {**empty, "face_counts": face_counts}

        # 逐对裁剪最大人脸区域并计算 SSIM
        pair_details = []
        pair_ssims = []
        for i in range(len(image_paths) - 1):
            face_a = all_faces[i][0]   # 最大脸（已按面积降序）
            face_b = all_faces[i + 1][0]

            crop_a = self._crop_face_region(image_paths[i], face_a)
            crop_b = self._crop_face_region(image_paths[i + 1], face_b)

            if crop_a is None or crop_b is None:
                pair_details.append({
                    'pair': f"{i + 1}-{i + 2}", 'face_ssim': 0.0, 'status': 'crop_failed'
                })
                pair_ssims.append(0.0)
                continue

            ssim_val = float(compare_ssim(crop_a, crop_b))
            pair_details.append({
                'pair': f"{i + 1}-{i + 2}",
                'face_ssim': round(ssim_val, 4),
            })
            pair_ssims.append(ssim_val)

        if not pair_ssims:
            return {**empty, "face_counts": face_counts}

        # 取最小 SSIM（最不相似的一对）作为判定基准
        min_ssim = min(pair_ssims)
        has_static = min_ssim >= self.static_face_ssim

        return {
            "face_static_score": round(min_ssim, 4) if has_static else 0.0,
            "pair_details": pair_details,
            "face_counts": face_counts,
            "has_static_face": has_static,
        }

    def compute_static_adjustment(self, face_static_score: float) -> float:
        """
        将人脸冻结分数映射为融合评分的上调量

        Args:
            face_static_score: 人脸冻结分数 (0.0 ~ 1.0)

        Returns:
            评分调整量（正数，用于加到融合评分上）
        """
        return face_static_score * self.static_face_adjustment

    # ---- 内部方法 ----

    def _detect_faces_in_image(self, image_path: str) -> List[Dict]:
        """
        对单张图片进行人脸检测

        Returns:
            人脸列表，每个元素为 dict: {center_x, center_y, width, height, area, confidence}
            坐标和尺寸均为相对值（0 ~ 1）
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning("FaceDetector: 无法读取图片 %s", image_path)
                return []

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with _mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=self.min_confidence
            ) as face_detection:
                results = face_detection.process(rgb)

            if not results.detections:
                return []

            faces = []
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                cx = bbox.xmin + bbox.width / 2.0
                cy = bbox.ymin + bbox.height / 2.0
                area = bbox.width * bbox.height
                faces.append({
                    'center_x': cx,
                    'center_y': cy,
                    'width': bbox.width,
                    'height': bbox.height,
                    'area': area,
                    'confidence': det.score[0] if det.score else 0.0,
                })

            # 按面积降序排列（大脸优先匹配）
            faces.sort(key=lambda f: f['area'], reverse=True)
            return faces

        except Exception as e:
            logger.warning("FaceDetector: 图片 %s 人脸检测失败: %s", image_path, e)
            return []

    def _crop_face_region(self, image_path: str, face: Dict, expand: float = 0.2) -> Optional[np.ndarray]:
        """
        从图片中裁剪人脸区域，返回统一尺寸的灰度图

        Args:
            image_path: 图片路径
            face: 人脸信息 dict（含相对坐标 center_x, center_y, width, height）
            expand: 向外扩展比例（0.2 = 四周各扩 20%），捕获更多上下文

        Returns:
            128x128 灰度 numpy array，失败返回 None
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            h, w = img.shape[:2]

            # 相对坐标 → 绝对像素坐标
            fw = face['width'] * w
            fh = face['height'] * h
            fx = (face['center_x'] - face['width'] / 2.0) * w
            fy = (face['center_y'] - face['height'] / 2.0) * h

            # 扩展 bbox
            ex = fw * expand
            ey = fh * expand
            x1 = max(int(fx - ex), 0)
            y1 = max(int(fy - ey), 0)
            x2 = min(int(fx + fw + ex), w)
            y2 = min(int(fy + fh + ey), h)

            if x2 <= x1 or y2 <= y1:
                return None

            crop = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            return resized

        except Exception as e:
            logger.warning("FaceDetector: 裁剪人脸区域失败 %s: %s", image_path, e)
            return None

    def _compare_pair(self, faces_a: List[Dict], faces_b: List[Dict], pair_idx: int) -> Dict:
        """
        比较相邻两帧的人脸变化

        比较维度：
        1. 数量变化：人脸数量是否改变
        2. 位置变化：匹配人脸的中心点偏移
        3. 大小变化：匹配人脸的面积变化

        Returns:
            dict: {pair, faces_a, faces_b, count_change, position_change, size_change, pair_score}
        """
        count_a = len(faces_a)
        count_b = len(faces_b)

        # 两帧都没有人脸 → 无法判断变化
        if count_a == 0 and count_b == 0:
            return {
                'pair': f"{pair_idx + 1}-{pair_idx + 2}",
                'faces_a': 0, 'faces_b': 0,
                'count_change': 0.0, 'position_change': 0.0, 'size_change': 0.0,
                'pair_score': 0.0,
            }

        # 只有一帧有人脸 → 强变化信号
        if count_a == 0 or count_b == 0:
            return {
                'pair': f"{pair_idx + 1}-{pair_idx + 2}",
                'faces_a': count_a, 'faces_b': count_b,
                'count_change': 1.0, 'position_change': 0.0, 'size_change': 0.0,
                'pair_score': 1.0,
            }

        # 数量变化分数
        count_change = abs(count_a - count_b) / max(count_a, count_b)

        # 匹配人脸并计算位置/大小变化
        position_change, size_change = self._match_and_compare(faces_a, faces_b)

        # 综合分数：取三个维度的最大值
        pair_score = max(count_change, position_change, size_change)

        return {
            'pair': f"{pair_idx + 1}-{pair_idx + 2}",
            'faces_a': count_a, 'faces_b': count_b,
            'count_change': round(count_change, 4),
            'position_change': round(position_change, 4),
            'size_change': round(size_change, 4),
            'pair_score': round(pair_score, 4),
        }

    def _match_and_compare(self, faces_a: List[Dict], faces_b: List[Dict]) -> Tuple[float, float]:
        """
        贪心匹配两组人脸，计算位置和大小变化

        匹配策略：对 faces_a 中的每张脸，在 faces_b 中找中心点距离最近的未匹配脸。

        Returns:
            (position_change_score, size_change_score)，均为 0.0 ~ 1.0
        """
        used_b = set()
        position_scores = []
        size_scores = []

        for fa in faces_a:
            best_dist = float('inf')
            best_idx = -1

            for j, fb in enumerate(faces_b):
                if j in used_b:
                    continue
                dist = math.hypot(fa['center_x'] - fb['center_x'], fa['center_y'] - fb['center_y'])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j

            if best_idx < 0:
                # 没有可匹配的脸 → 视为完全变化
                position_scores.append(1.0)
                size_scores.append(1.0)
                continue

            used_b.add(best_idx)
            fb = faces_b[best_idx]

            # 位置变化：中心点欧氏距离 vs 阈值，归一化到 [0, 1]
            pos_score = min(best_dist / self.position_threshold, 1.0) if self.position_threshold > 0 else 0.0
            position_scores.append(pos_score)

            # 大小变化：面积相对差 vs 阈值，归一化到 [0, 1]
            area_diff = abs(fa['area'] - fb['area']) / max(fa['area'], fb['area'], 1e-6)
            sz_score = min(area_diff / self.size_threshold, 1.0) if self.size_threshold > 0 else 0.0
            size_scores.append(sz_score)

        # 取所有匹配对的最大值（最显著的变化）
        position_change = max(position_scores) if position_scores else 0.0
        size_change = max(size_scores) if size_scores else 0.0

        return position_change, size_change
