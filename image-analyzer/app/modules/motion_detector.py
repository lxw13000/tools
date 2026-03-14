"""
图片静态/动态检测模块
通过比较连续图片的感知哈希(pHash)相似度来判断是否为动态内容
"""

import numpy as np
import imagehash
from PIL import Image
from typing import List, Dict
import time


class MotionDetector:
    """图片序列动态检测器"""

    def __init__(self, similarity_threshold: float = 0.95):
        """
        初始化动态检测器

        Args:
            similarity_threshold: 相似度阈值，高于此值认为是静态图片
        """
        self.similarity_threshold = similarity_threshold

    def detect(self, image_paths: List[str]) -> Dict:
        """
        检测图片序列是静态还是动态

        Args:
            image_paths: 图片路径列表

        Returns:
            dict: {status, result('static'|'dynamic'), confidence, similarities, message, elapsed_seconds}
                  失败时返回 {status:'error', message}
        """
        if not image_paths:
            return {
                "status": "error",
                "message": "没有提供图片"
            }

        start_time = time.time()

        if len(image_paths) == 1:
            elapsed = round(time.time() - start_time, 2)
            return {
                "status": "success",
                "result": "static",
                "confidence": 1.0,
                "message": "只有一张图片，判定为静态",
                "elapsed_seconds": elapsed
            }

        try:
            # 计算连续图片之间的相似度
            similarities = self._calculate_similarities(image_paths)

            if not similarities:
                return {
                    "status": "error",
                    "message": "无法计算图片相似度（有效图片不足 2 张）"
                }

            # 计算平均相似度
            avg_similarity = np.mean(similarities)

            # 判断是静态还是动态
            is_static = avg_similarity >= self.similarity_threshold

            elapsed = round(time.time() - start_time, 2)

            return {
                "status": "success",
                "result": "static" if is_static else "dynamic",
                "confidence": float(avg_similarity),
                "similarities": [float(s) for s in similarities],
                "message": f"平均相似度: {avg_similarity:.2%}",
                "elapsed_seconds": elapsed
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"动态检测失败: {str(e)}"
            }

    def _calculate_similarities(self, image_paths: List[str]) -> List[float]:
        """
        计算连续图片之间的相似度

        Args:
            image_paths: 图片路径列表

        Returns:
            相似度列表；有效图片 < 2 张时返回空列表
        """
        # 逐图计算 pHash，跳过损坏/无法读取的图片
        hashes = []
        for path in image_paths:
            try:
                img = Image.open(path)
                img_hash = imagehash.phash(img)
                hashes.append(img_hash)
            except Exception as e:
                print(f"跳过无法读取的图片 {path}: {e}")
                continue

        if len(hashes) < 2:
            return []

        similarities = []
        # 计算相邻图片的哈希差异
        for i in range(len(hashes) - 1):
            hash_diff = hashes[i] - hashes[i + 1]
            # pHash 差异范围 0-64（64 位），转换为 0-1 相似度：1 表示完全相同
            similarity = 1 - (hash_diff / 64.0)
            similarities.append(similarity)

        return similarities
