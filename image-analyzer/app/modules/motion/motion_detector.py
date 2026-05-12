"""
图片静态/动态检测模块

核心业务：传入 1-6 张图片，通过三种方法（pHash、SSIM、光流法）综合评分，
判断本次检测的图片序列属于高风险挂播、中风险挂播、人工复核还是通过。

三种评分方法：
  - pHash（感知哈希）：比较 64 位感知哈希的汉明距离，快速粗判
  - SSIM（结构相似度）：基于亮度/对比度/结构的图像质量指标
  - Optical Flow（光流法）：基于 Farneback 稠密光流的运动检测

融合逻辑：
  - 各方法计算相邻帧相似度（0-1），取平均后按权重加权融合
  - 权重自动归一化，某方法失败时自动排除并重新分配

判定逻辑（三阈值四分类）：
  - 1 张图片 → 拒绝检测（至少需要 2 张）
  - 2-6 张图片 → 计算融合评分与三阈值比较
    - 融合评分 >= 高风险阈值 → 高风险挂播
    - 融合评分 >= 中风险阈值 → 中风险挂播
    - 融合评分 >= 复核阈值   → 人工复核
    - 融合评分 <  复核阈值   → 通过

返回结果 status 取值：
  - 'success' — 检测成功，result 为 'high_risk'、'mid_risk'、'review' 或 'pass'
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

from .face_detector import FaceDetector

logger = logging.getLogger(__name__)

# 单次检测允许的最大图片数量（业务规则：1-6 张）
MAX_IMAGES = 6


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
            'phash': float(weights.get('phash', 0.25)),
            'ssim': float(weights.get('ssim', 0.35)),
            'flow': float(weights.get('flow', 0.40)),
        }
        # 三阈值四分类（高风险 > 中风险 > 复核，低于复核阈值为通过）
        self.thresholds = {
            'high_risk': float(thresholds.get('high_risk', 0.95)),
            'mid_risk': float(thresholds.get('mid_risk', 0.87)),
            'review': float(thresholds.get('review', 0.78)),
        }
        # 算法调优参数
        self.target_size = int(config.get('target_size', 512))
        self.phash_hash_size = int(config.get('phash_hash_size', 16))
        self.min_weight = float(config.get('min_weight', 0.4))
        self.flow_motion_threshold = float(config.get('flow_motion_threshold', 1.5))
        self.flow_scale_factor = float(config.get('flow_scale_factor', 6.0))
        # CLAHE 亮度归一化增强配置（高光线截图检测）
        clahe_config = config.get('clahe_enhancement', {})
        self.clahe_enabled = bool(clahe_config.get('enabled', True))
        self.clahe_clip_limit = float(clahe_config.get('clip_limit', 2.0))
        self.clahe_tile_size = int(clahe_config.get('tile_size', 8))
        self.clahe_gap_threshold = float(clahe_config.get('gap_threshold', 0.08))
        self.clahe_adjustment_factor = float(clahe_config.get('adjustment_factor', 1.5))
        self.clahe_max_adjustment = float(clahe_config.get('max_adjustment', 0.25))
        # 合成挂播检测配置（静态图+动态特效检测，像素级静态比率）
        block_config = config.get('block_static', {})
        self.block_static_enabled = bool(block_config.get('enabled', True))
        self.block_pixel_diff_threshold = int(block_config.get('pixel_diff_threshold', 15))
        self.block_min_static_ratio = float(block_config.get('min_static_ratio', 0.75))
        self.block_static_adjustment = float(block_config.get('adjustment', 0.50))
        # 人脸检测增强（可选）
        face_config = config.get('face_detection', {})
        self.face_detector = FaceDetector(config=face_config)

    def detect(self, image_paths: List[str],
               weights: Optional[Dict] = None,
               thresholds: Optional[Dict] = None,
               face_detection_enabled: Optional[bool] = None) -> Dict:
        """
        检测图片序列是静态、动态还是需要人工复核

        Args:
            image_paths: 图片文件路径列表（2-6 张）
            weights:     可选的权重覆盖（前端传入），如 {'phash': 0.5, 'ssim': 0.3, 'flow': 0.2}
            thresholds:  可选的阈值覆盖（前端传入），如 {'high_risk': 0.95, 'mid_risk': 0.85, 'review': 0.70}
            face_detection_enabled: 可选的人脸检测开关覆盖（None 表示使用配置默认值）

        Returns:
            dict: 检测结果
                成功时: {status, result, result_text, fusion_score, scores, weights_used,
                         thresholds_used, pair_details, message, elapsed_seconds,
                         [face_detection_used, face_change_score, face_details, original_fusion_score]}
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
        overall_start = time.perf_counter()
        stage_ms = {}   # 各阶段毫秒耗时统计，用于日志和返回

        # 合并权重和阈值（前端覆盖 > 实例默认值）
        w = {**self.weights, **(weights or {})}
        t = {**self.thresholds, **(thresholds or {})}

        result_map = {'high_risk': '高风险挂播', 'mid_risk': '中风险挂播', 'review': '人工复核', 'pass': '通过'}

        # ---- 单张图片无法比较，拒绝检测 ----
        if len(image_paths) < 2:
            return {
                "status": "error",
                "message": "至少需要 2 张图片才能检测动静态，当前仅 1 张",
            }

        # ---- 多张图片：三维融合评分 ----
        try:
            # 计算各方法的相邻帧相似度
            _t = time.perf_counter()
            phash_sims = self._calc_phash_similarities(image_paths)
            stage_ms['phash'] = int(round((time.perf_counter() - _t) * 1000))

            _t = time.perf_counter()
            ssim_sims = self._calc_ssim_similarities(image_paths)
            stage_ms['ssim'] = int(round((time.perf_counter() - _t) * 1000))

            _t = time.perf_counter()
            flow_sims = self._calc_flow_similarities(image_paths)
            stage_ms['flow'] = int(round((time.perf_counter() - _t) * 1000))

            # 各方法取聚合值（min-weighted：兼顾最差对与均值）
            phash_avg = self._aggregate_pairs(phash_sims) if phash_sims else None
            ssim_avg = self._aggregate_pairs(ssim_sims) if ssim_sims else None
            flow_avg = self._aggregate_pairs(flow_sims) if flow_sims else None

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

            # ---- 四态判定 ----
            high_risk_th = t['high_risk']
            mid_risk_th = t['mid_risk']
            review_th = t['review']

            if final_score >= high_risk_th:
                result = 'high_risk'
            elif final_score >= mid_risk_th:
                result = 'mid_risk'
            elif final_score >= review_th:
                result = 'review'
            else:
                result = 'pass'

            # ---- 人脸检测增强（可选后处理）----
            face_result = None
            original_score = final_score

            # 确定本次请求是否启用人脸检测
            use_face = (face_detection_enabled
                        if face_detection_enabled is not None
                        else self.face_detector.is_available())

            if use_face and result in ('high_risk', 'mid_risk', 'review'):
                try:
                    _t = time.perf_counter()
                    face_result = self.face_detector.detect_face_changes(image_paths)
                    stage_ms['face_change'] = int(round((time.perf_counter() - _t) * 1000))
                    if face_result.get('face_change_score', 0) > 0:
                        adjustment = self.face_detector.compute_adjustment(
                            face_result['face_change_score']
                        )
                        final_score = round(max(final_score - adjustment, 0.0), 4)

                        # 以调整后评分重新分类
                        if final_score >= high_risk_th:
                            result = 'high_risk'
                        elif final_score >= mid_risk_th:
                            result = 'mid_risk'
                        elif final_score >= review_th:
                            result = 'review'
                        else:
                            result = 'pass'

                        logger.info("人脸检测增强: 变化度=%.4f, 调整量=%.4f, 评分 %.4f→%.4f, 判定 %s",
                                    face_result['face_change_score'], adjustment,
                                    original_score, final_score, result)
                except Exception as e:
                    logger.warning("人脸检测增强失败，使用原始评分: %s", e)
                    face_result = None

            # ---- 合成挂播检测（人脸冻结 + 背景动态 = 疑似合成）----
            if use_face and result == 'pass':
                try:
                    _t = time.perf_counter()
                    static_result = self.face_detector.detect_static_faces(image_paths)
                    stage_ms['face_static'] = int(round((time.perf_counter() - _t) * 1000))
                    if static_result.get('has_static_face'):
                        adjustment = self.face_detector.compute_static_adjustment(
                            static_result['face_static_score']
                        )
                        final_score = round(min(final_score + adjustment, 1.0), 4)

                        # 以调整后评分重新分类
                        if final_score >= high_risk_th:
                            result = 'high_risk'
                        elif final_score >= mid_risk_th:
                            result = 'mid_risk'
                        elif final_score >= review_th:
                            result = 'review'
                        else:
                            result = 'pass'

                        face_result = static_result
                        face_result['composite_detected'] = True
                        logger.info("合成挂播检测: 人脸冻结度=%.4f, 调整量=+%.4f, 评分 %.4f→%.4f, 判定 %s",
                                    static_result['face_static_score'], adjustment,
                                    original_score, final_score, result)
                except Exception as e:
                    logger.warning("合成挂播检测失败: %s", e)

            # ---- CLAHE 亮度归一化重评估（高光线截图检测）----
            clahe_result = None
            if self.clahe_enabled and result == 'pass':
                try:
                    _t = time.perf_counter()
                    clahe_ssims = self._calc_clahe_ssim_similarities(image_paths)
                    stage_ms['clahe'] = int(round((time.perf_counter() - _t) * 1000))
                    if clahe_ssims and ssim_sims:
                        clahe_avg = self._aggregate_pairs(clahe_ssims)
                        ssim_orig_avg = self._aggregate_pairs(ssim_sims)
                        gap = clahe_avg - ssim_orig_avg
                        if gap > self.clahe_gap_threshold:
                            adjustment = min(gap * self.clahe_adjustment_factor,
                                             self.clahe_max_adjustment)
                            final_score = round(min(final_score + adjustment, 1.0), 4)

                            if final_score >= high_risk_th:
                                result = 'high_risk'
                            elif final_score >= mid_risk_th:
                                result = 'mid_risk'
                            elif final_score >= review_th:
                                result = 'review'
                            else:
                                result = 'pass'

                            clahe_result = {
                                'clahe_gap': round(gap, 4),
                                'clahe_ssim_avg': round(clahe_avg, 4),
                                'adjustment': round(adjustment, 4),
                            }
                            logger.info("CLAHE增强: gap=%.4f, 调整量=+%.4f, 评分 %.4f→%.4f, 判定 %s",
                                        gap, adjustment, original_score, final_score, result)
                except Exception as e:
                    logger.warning("CLAHE增强失败: %s", e)

            # ---- 分块静态比率检测（静态图+动态特效检测）----
            block_result = None
            if self.block_static_enabled and result == 'pass':
                try:
                    _t = time.perf_counter()
                    block_ratios = self._calc_block_static_ratio(image_paths)
                    stage_ms['block_static'] = int(round((time.perf_counter() - _t) * 1000))
                    if block_ratios:
                        min_ratio = min(block_ratios)
                        if min_ratio > self.block_min_static_ratio:
                            adjustment = min_ratio * self.block_static_adjustment
                            final_score = round(min(final_score + adjustment, 1.0), 4)

                            if final_score >= high_risk_th:
                                result = 'high_risk'
                            elif final_score >= mid_risk_th:
                                result = 'mid_risk'
                            elif final_score >= review_th:
                                result = 'review'
                            else:
                                result = 'pass'

                            block_result = {
                                'min_static_ratio': round(min_ratio, 4),
                                'pair_ratios': [round(r, 4) for r in block_ratios],
                                'adjustment': round(adjustment, 4),
                            }
                            logger.info("分块静态检测: min_ratio=%.4f, 调整量=+%.4f, 评分 %.4f→%.4f, 判定 %s",
                                        min_ratio, adjustment, original_score, final_score, result)
                except Exception as e:
                    logger.warning("分块静态检测失败: %s", e)

            elapsed = round(time.time() - start_time, 2)
            elapsed_ms = int(round((time.perf_counter() - overall_start) * 1000))
            logger.info("动态检测完成: result=%s, fusion_score=%.4f, elapsed=%dms, stages=%s",
                        result, final_score, elapsed_ms, stage_ms)

            result_dict = {
                "status": "success",
                "result": result,
                "result_text": result_map[result],
                "fusion_score": final_score,
                "scores": scores_detail,
                "weights_used": weights_used,
                "thresholds_used": {"high_risk": high_risk_th, "mid_risk": mid_risk_th, "review": review_th},
                "pair_details": {
                    "phash": [round(s, 4) for s in phash_sims],
                    "ssim": [round(s, 4) for s in ssim_sims],
                    "flow": [round(s, 4) for s in flow_sims],
                },
                "message": f"融合评分: {final_score:.2%}, 判定: {result_map[result]}",
                "elapsed_seconds": elapsed,
                "elapsed_ms": elapsed_ms,
                "stage_ms": stage_ms,
            }

            # 人脸检测信息仅在实际使用时附加（向后兼容）
            if face_result is not None:
                result_dict["face_detection_used"] = True
                result_dict["face_details"] = face_result.get("pair_details", [])
                result_dict["face_counts"] = face_result.get("face_counts", [])
                result_dict["original_fusion_score"] = original_score
                if face_result.get('composite_detected'):
                    # 合成挂播检测结果
                    result_dict["composite_detected"] = True
                    result_dict["face_static_score"] = face_result.get("face_static_score", 0.0)
                else:
                    # 常规人脸变化检测结果
                    result_dict["face_change_score"] = face_result.get("face_change_score", 0.0)

            # CLAHE 增强信息仅在触发时附加
            if clahe_result is not None:
                result_dict["clahe_enhanced"] = True
                result_dict["clahe_gap"] = clahe_result['clahe_gap']
                result_dict["original_fusion_score"] = result_dict.get("original_fusion_score", original_score)

            # 分块静态检测信息仅在触发时附加
            if block_result is not None:
                result_dict["block_static_detected"] = True
                result_dict["block_static_ratio"] = block_result['min_static_ratio']
                result_dict["original_fusion_score"] = result_dict.get("original_fusion_score", original_score)

            return result_dict

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
                    img_hash = imagehash.phash(img, hash_size=self.phash_hash_size)
                    hashes.append(img_hash)
            except Exception as e:
                logger.warning("pHash: 跳过无法读取的图片 %s: %s", path, e)
                continue

        if len(hashes) < 2:
            return []

        similarities = []
        max_distance = float(self.phash_hash_size ** 2)
        for i in range(len(hashes) - 1):
            hash_diff = hashes[i] - hashes[i + 1]
            similarity = 1.0 - (hash_diff / max_distance)
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
                    gray = img.convert('L').resize((self.target_size, self.target_size), Image.Resampling.BILINEAR)
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
                with Image.open(path) as img:
                    gray = img.convert('L').resize(
                        (self.target_size, self.target_size), Image.Resampling.BILINEAR
                    )
                    grays.append(np.array(gray))
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
            # 运动像素占比（替代旧版 mean_magnitude，不被静态背景稀释）
            motion_mask = magnitude > self.flow_motion_threshold
            motion_ratio = float(np.sum(motion_mask)) / float(magnitude.size)
            similarity = max(1.0 - motion_ratio * self.flow_scale_factor, 0.0)
            similarities.append(similarity)

        return similarities

    # ---- 辅助方法：逐对相似度聚合 ----

    def _aggregate_pairs(self, similarities: List[float]) -> float:
        """
        聚合逐对相似度分数

        使用最小值权重策略：min_weight * min(pairs) + (1 - min_weight) * avg(pairs)，
        确保单对运动信号不被 N-1 对静态分数淹没。

        min_weight=0.0 时退化为纯平均（向后兼容）。
        """
        if not similarities:
            return 0.0
        avg = sum(similarities) / len(similarities)
        if self.min_weight <= 0.0 or len(similarities) == 1:
            return avg
        min_val = min(similarities)
        return self.min_weight * min_val + (1.0 - self.min_weight) * avg

    # ---- 增强方法 1: CLAHE 亮度归一化 SSIM（高光线截图检测）----

    def _calc_clahe_ssim_similarities(self, image_paths: List[str]) -> List[float]:
        """
        CLAHE 亮度归一化后计算 SSIM

        对图片做自适应直方图均衡化（CLAHE）后再计算 SSIM，
        消除高光/反光/亮度波动的影响。若 CLAHE 后 SSIM 显著高于
        原始 SSIM，说明原始差异主要来自亮度而非内容变化。

        Args:
            image_paths: 图片路径列表

        Returns:
            CLAHE 处理后的 SSIM 相似度列表；有效图片 < 2 张时返回空列表
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
        )
        grays = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    gray = np.array(img.convert('L').resize(
                        (self.target_size, self.target_size), Image.Resampling.BILINEAR
                    ))
                    gray = clahe.apply(gray)
                    grays.append(gray)
            except Exception as e:
                logger.warning("CLAHE: 跳过无法读取的图片 %s: %s", path, e)
                continue

        if len(grays) < 2:
            return []

        return [float(compare_ssim(grays[i], grays[i + 1]))
                for i in range(len(grays) - 1)]

    # ---- 增强方法 2: 像素级静态比率检测（静态图+动态特效检测）----

    def _calc_block_static_ratio(self, image_paths: List[str]) -> List[float]:
        """
        计算像素级静态比率

        逐对比较相邻帧的灰度像素差值，统计差值低于阈值的"静态像素"占比。
        静态图片+动态特效合成的特征：底层大部分像素不变（差值极小），
        仅特效覆盖区域有像素变化。像素级检测不受特效大小影响，
        比分块 SSIM 更精准。

        Args:
            image_paths: 图片路径列表

        Returns:
            每对相邻图片的静态像素占比列表（0-1）；有效图片 < 2 张时返回空列表
        """
        grays = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    gray = np.array(img.convert('L').resize(
                        (self.target_size, self.target_size), Image.Resampling.BILINEAR
                    ))
                    grays.append(gray)
            except Exception as e:
                logger.warning("PixelStatic: 跳过无法读取的图片 %s: %s", path, e)
                continue

        if len(grays) < 2:
            return []

        ratios = []
        for i in range(len(grays) - 1):
            diff = np.abs(grays[i].astype(np.int16) - grays[i + 1].astype(np.int16))
            static_pixels = int(np.sum(diff < self.block_pixel_diff_threshold))
            ratio = static_pixels / diff.size
            ratios.append(ratio)

        return ratios
