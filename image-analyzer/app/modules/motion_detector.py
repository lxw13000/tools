"""
图片静态/动态检测模块
通过比较连续图片的相似度来判断是否为动态内容
"""

import cv2
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
            包含检测结果的字典
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

        # 计算连续图片之间的相似度
        similarities = self._calculate_similarities(image_paths)

        if not similarities:
            return {
                "status": "error",
                "message": "无法计算图片相似度"
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

    def _calculate_similarities(self, image_paths: List[str]) -> List[float]:
        """
        计算连续图片之间的相似度

        Args:
            image_paths: 图片路径列表

        Returns:
            相似度列表
        """
        similarities = []

        try:
            # 使用感知哈希算法比较图片
            hashes = []
            for path in image_paths:
                img = Image.open(path)
                img_hash = imagehash.phash(img)
                hashes.append(img_hash)

            # 计算相邻图片的哈希差异
            for i in range(len(hashes) - 1):
                hash_diff = hashes[i] - hashes[i + 1]
                # 将哈希差异转换为相似度 (0-1)
                # phash返回的差异范围是0-64，差异越小越相似
                similarity = 1 - (hash_diff / 64.0)
                similarities.append(similarity)

        except Exception as e:
            print(f"计算相似度时出错: {str(e)}")
            return []

        return similarities

    def _calculate_ssim(self, img1_path: str, img2_path: str) -> float:
        """
        使用结构相似性指数(SSIM)计算两张图片的相似度
        备用方法，更精确但计算量更大

        Args:
            img1_path: 第一张图片路径
            img2_path: 第二张图片路径

        Returns:
            相似度值 (0-1)
        """
        try:
            # 读取图片
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                return 0.0

            # 调整图片大小使其一致
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # 计算SSIM
            from skimage.metrics import structural_similarity
            ssim_value = structural_similarity(img1, img2)

            return float(ssim_value)

        except Exception as e:
            print(f"计算SSIM时出错: {str(e)}")
            return 0.0
