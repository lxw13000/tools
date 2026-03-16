"""
图片静态/动态检测模块

核心业务：传入 1-6 张图片，通过比较连续图片的感知哈希(pHash)相似度，
判断本次检测的图片序列属于静态画面还是动态画面。

判定逻辑：
  - 1 张图片 → 直接判定为静态（无比较对象）
  - 2-6 张图片 → 计算相邻图片 pHash 相似度，取平均值与阈值比较
    - 平均相似度 >= 阈值 → 静态
    - 平均相似度 <  阈值 → 动态

返回结果 status 取值：
  - 'success' — 检测成功，result 为 'static' 或 'dynamic'
  - 'error'   — 检测失败（图片数量不合法、图片读取失败等）
"""

import imagehash
from PIL import Image
from typing import List, Dict
import time
import logging

logger = logging.getLogger(__name__)

# 单次检测允许的最大图片数量（业务规则：1-6 张）
MAX_IMAGES = 6


class MotionDetector:
    """图片序列动态检测器：基于 pHash 相似度判断静态/动态"""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        初始化动态检测器

        Args:
            similarity_threshold: 相似度阈值（0-1），高于此值判定为静态，默认 0.95
        """
        self.similarity_threshold = similarity_threshold

    def detect(self, image_paths: List[str]) -> Dict:
        """
        检测图片序列是静态还是动态

        Args:
            image_paths: 图片文件路径列表（1-6 张）

        Returns:
            dict: 检测结果
                成功时: {status:'success', result:'static'|'dynamic', confidence, similarities, message, elapsed_seconds}
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

        # ---- 单张图片直接判定为静态 ----
        if len(image_paths) == 1:
            elapsed = round(time.time() - start_time, 2)
            return {
                "status": "success",
                "result": "static",
                "confidence": 1.0,
                "message": "只有一张图片，判定为静态",
                "elapsed_seconds": elapsed,
            }

        # ---- 多张图片：计算相邻帧 pHash 相似度 ----
        try:
            similarities = self._calculate_similarities(image_paths)

            if not similarities:
                return {
                    "status": "error",
                    "message": "无法计算图片相似度（有效图片不足 2 张）",
                }

            # 取所有相邻帧相似度的算术平均值作为整体置信度
            avg_similarity = sum(similarities) / len(similarities)

            # 平均相似度 >= 阈值 → 静态，否则 → 动态
            is_static = avg_similarity >= self.similarity_threshold

            elapsed = round(time.time() - start_time, 2)

            return {
                "status": "success",
                "result": "static" if is_static else "dynamic",
                "confidence": round(float(avg_similarity), 4),
                "similarities": [round(float(s), 4) for s in similarities],
                "message": f"平均相似度: {avg_similarity:.2%}",
                "elapsed_seconds": elapsed,
            }

        except Exception as e:
            logger.exception("动态检测失败")
            return {"status": "error", "message": f"动态检测失败: {str(e)}"}

    def _calculate_similarities(self, image_paths: List[str]) -> List[float]:
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
                # 使用 with 确保文件句柄及时释放
                with Image.open(path) as img:
                    img_hash = imagehash.phash(img)
                    hashes.append(img_hash)
            except Exception as e:
                # 跳过损坏或无法读取的图片，记录日志
                logger.warning("跳过无法读取的图片 %s: %s", path, e)
                continue

        # 有效图片不足 2 张，无法计算相似度
        if len(hashes) < 2:
            return []

        similarities = []
        for i in range(len(hashes) - 1):
            # pHash 汉明距离：0 表示完全相同，64 表示完全不同
            hash_diff = hashes[i] - hashes[i + 1]
            # 归一化为相似度：1.0 = 完全相同，0.0 = 完全不同
            similarity = 1.0 - (hash_diff / 64.0)
            similarities.append(similarity)

        return similarities
