"""
图片静态/动态检测模块

核心业务：传入 1-6 张图片，通过三种方法（pHash、SSIM、光流法）综合评分，
判断本次检测的图片序列属于静态画面、动态画面还是需要人工复核。

三种评分方法：
  - pHash（感知哈希）：比较 64 位感知哈希的汉明距离，快速粗判
  - SSIM（结构相似度）：基于亮度/对比度/结构的图像质量指标
  - Optical Flow（光流法）：基于 Farneback 稠密光流的运动检测

融合逻辑：
  - 各方法计算相邻帧相似度（0-1），取平均后按权重加权融合
  - 权重自动归一化，某方法失败时自动排除并重新分配

判定逻辑：
  - 1 张图片 → 直接判定为静态（无比较对象）
  - 2-6 张图片 → 计算融合评分与双阈值比较
    - 融合评分 >= 静态阈值 → 静态
    - 融合评分 <  动态阈值 → 动态
    - 介于两者之间 → 人工复核

返回结果 status 取值：
  - 'success' — 检测成功，result 为 'static'、'dynamic' 或 'review'
  - 'error'   — 检测失败（图片数量不合法、图片读取失败等）
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from typing import List, Dict, Optional
import time
import logging

logger = logging.getLogger(__name__)

# 单次检测允许的最大图片数量（业务规则：1-6 张）
MAX_IMAGES = 6

# SSIM 和光流计算时统一缩放的目标尺寸（平衡精度与性能）
TARGET_SIZE = (256, 256)


class MotionDetector:
    """图片序列动态检测器：基于 pHash + SSIM + 光流法三维融合评分"""

    def __init__(self, config: dict = None):
        """
        初始化动态检测器

        Args:
            config: motion_detection 配置字典，包含 weights 和 thresholds 子项
        """
        config = config or {}
        weights = config.get('weights', {})
        thresholds = config.get('thresholds', {})

        # 三种评分方法的默认权重
        self.weights = {
            'phash': float(weights.get('phash', 0.40)),
            'ssim': float(weights.get('ssim', 0.35)),
            'flow': float(weights.get('flow', 0.25)),
        }
        # 双阈值（静态阈值 > 动态阈值，中间区域为人工复核）
        self.thresholds = {
            'static': float(thresholds.get('static', 0.95)),
            'dynamic': float(thresholds.get('dynamic', 0.75)),
        }

    def detect(self, image_paths: List[str],
               weights: Optional[Dict] = None,
               thresholds: Optional[Dict] = None) -> Dict:
        """
        检测图片序列是静态、动态还是需要人工复核

        Args:
            image_paths: 图片文件路径列表（1-6 张）
            weights:     可选的权重覆盖（前端传入），如 {'phash': 0.5, 'ssim': 0.3, 'flow': 0.2}
            thresholds:  可选的阈值覆盖（前端传入），如 {'static': 0.90, 'dynamic': 0.70}

        Returns:
            dict: 检测结果
                成功时: {status, result, result_text, fusion_score, scores, weights_used,
                         thresholds_used, pair_details, message, elapsed_seconds}
                失败时: {status:'error', message}
        """
        # ---- 参数校验 ----
        if not image_paths:
            return {"status": "error", "message": "没有提供图片"}

        if len(image_paths) > MAX_IMAGES:
            return {
                "status": "error",
                "message": f"图片数量超出限制，最多支持 {MAX_IMAGES} 张，当前 {len(image_paths)} 张",
            }

        start_time = time.time()

        # 合并权重和阈值（前端覆盖 > 实例默认值）
        w = {**self.weights, **(weights or {})}
        t = {**self.thresholds, **(thresholds or {})}

        result_map = {'static': '静态', 'dynamic': '动态', 'review': '人工复核'}

        # ---- 单张图片直接判定为静态 ----
        if len(image_paths) == 1:
            elapsed = round(time.time() - start_time, 2)
            return {
                "status": "success",
                "result": "static",
                "result_text": "静态",
                "fusion_score": 1.0,
                "scores": {},
                "weights_used": w,
                "thresholds_used": t,
                "pair_details": {"phash": [], "ssim": [], "flow": []},
                "message": "只有一张图片，判定为静态",
                "elapsed_seconds": elapsed,
            }

        # ---- 多张图片：三维融合评分 ----
        try:
            # 计算各方法的相邻帧相似度
            phash_sims = self._calc_phash_similarities(image_paths)
            ssim_sims = self._calc_ssim_similarities(image_paths)
            flow_sims = self._calc_flow_similarities(image_paths)

            # 各方法取平均
            phash_avg = sum(phash_sims) / len(phash_sims) if phash_sims else None
            ssim_avg = sum(ssim_sims) / len(ssim_sims) if ssim_sims else None
            flow_avg = sum(flow_sims) / len(flow_sims) if flow_sims else None

            # 权重归一化并加权融合（自动排除失败的方法）
            active_score = 0.0
            active_weight = 0.0
            scores_detail = {}

            for name, avg, weight in [
                ('phash', phash_avg, w['phash']),
                ('ssim', ssim_avg, w['ssim']),
                ('flow', flow_avg, w['flow']),
            ]:
                if avg is not None:
                    active_score += avg * weight
                    active_weight += weight
                    scores_detail[name] = round(avg, 4)

            if active_weight == 0:
                return {"status": "error", "message": "所有评分方法均失败，无法计算融合评分"}

            # 归一化（处理权重和不等于 1 或部分方法失败的情况）
            final_score = round(active_score / active_weight, 4)

            # 归一化后的实际权重
            weights_used = {}
            for name, weight in [('phash', w['phash']), ('ssim', w['ssim']), ('flow', w['flow'])]:
                if name in scores_detail:
                    weights_used[name] = round(weight / active_weight, 4)
                else:
                    weights_used[name] = 0.0

            # ---- 三态判定 ----
            static_th = t['static']
            dynamic_th = t['dynamic']

            if final_score >= static_th:
                result = 'static'
            elif final_score < dynamic_th:
                result = 'dynamic'
            else:
                result = 'review'

            elapsed = round(time.time() - start_time, 2)

            return {
                "status": "success",
                "result": result,
                "result_text": result_map[result],
                "fusion_score": final_score,
                "scores": scores_detail,
                "weights_used": weights_used,
                "thresholds_used": {"static": static_th, "dynamic": dynamic_th},
                "pair_details": {
                    "phash": [round(s, 4) for s in phash_sims],
                    "ssim": [round(s, 4) for s in ssim_sims],
                    "flow": [round(s, 4) for s in flow_sims],
                },
                "message": f"融合评分: {final_score:.2%}, 判定: {result_map[result]}",
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            logger.exception("动态检测失败")
            return {"status": "error", "message": f"动态检测失败: {str(e)}"}

    # ---- 评分方法 1: pHash（感知哈希）----

    def _calc_phash_similarities(self, image_paths: List[str]) -> List[float]:
        """
        计算连续图片之间的 pHash 相似度

        对每张图片计算 64 位感知哈希，再逐对比较相邻帧的汉明距离，
        将距离归一化为 0-1 的相似度（1 = 完全相同，0 = 完全不同）。

        Args:
            image_paths: 图片路径列表

        Returns:
            相似度列表（长度 = 图片数 - 1）；有效图片 < 2 张时返回空列表
        """
        hashes = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img_hash = imagehash.phash(img)
                    hashes.append(img_hash)
            except Exception as e:
                logger.warning("pHash: 跳过无法读取的图片 %s: %s", path, e)
                continue

        if len(hashes) < 2:
            return []

        similarities = []
        for i in range(len(hashes) - 1):
            hash_diff = hashes[i] - hashes[i + 1]
            similarity = 1.0 - (hash_diff / 64.0)
            similarities.append(similarity)

        return similarities

    # ---- 评分方法 2: SSIM（结构相似度）----

    def _calc_ssim_similarities(self, image_paths: List[str]) -> List[float]:
        """
        计算连续图片之间的 SSIM 相似度

        将图片转为灰度并统一缩放至 256x256，使用 skimage 的 structural_similarity 计算。
        SSIM 基于亮度、对比度、结构三个维度衡量图像相似性，对局部结构变化敏感。

        Args:
            image_paths: 图片路径列表

        Returns:
            相似度列表（0-1，1 = 完全相同）；有效图片 < 2 张时返回空列表
        """
        grays = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    gray = img.convert('L').resize(TARGET_SIZE, Image.Resampling.BILINEAR)
                    grays.append(np.array(gray))
            except Exception as e:
                logger.warning("SSIM: 跳过无法读取的图片 %s: %s", path, e)
                continue

        if len(grays) < 2:
            return []

        similarities = []
        for i in range(len(grays) - 1):
            score = compare_ssim(grays[i], grays[i + 1])
            similarities.append(float(score))

        return similarities

    # ---- 评分方法 3: Optical Flow（光流法）----

    def _calc_flow_similarities(self, image_paths: List[str]) -> List[float]:
        """
        计算连续图片之间的光流相似度

        使用 Farneback 稠密光流算法计算相邻帧之间的像素级运动向量，
        取运动幅值的均值并归一化为相似度（无运动 → 1，运动越大 → 趋近 0）。

        归一化公式：similarity = 1.0 / (1.0 + mean_magnitude)

        Args:
            image_paths: 图片路径列表

        Returns:
            相似度列表（0-1，1 = 无运动）；有效图片 < 2 张时返回空列表
        """
        grays = []
        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    logger.warning("Flow: 无法读取图片 %s", path)
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, TARGET_SIZE)
                grays.append(gray)
            except Exception as e:
                logger.warning("Flow: 跳过无法读取的图片 %s: %s", path, e)
                continue

        if len(grays) < 2:
            return []

        similarities = []
        for i in range(len(grays) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                grays[i], grays[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_mag = float(np.mean(magnitude))
            similarity = 1.0 / (1.0 + mean_mag)
            similarities.append(similarity)

        return similarities
