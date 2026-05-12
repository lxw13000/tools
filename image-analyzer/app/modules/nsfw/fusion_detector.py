"""
多模型融合决策引擎

通过 NSFWDetector 门面调用多个检测模型，加权融合各模型的安全分类分数，
综合「色情」和「性感」两个维度做出最终安全判定。

融合流程：
  1. 依次调用所选模型的 detect()，收集每个模型的色情和性感分数
  2. 按模型权重加权平均，得到融合色情分数(final_porn)和融合性感分数(final_sexy)
  3. 计算综合不安全分数 = final_porn + final_sexy（性感参与阈值评判）
  4. 根据决策策略和阈值，输出最终 action

支持三种决策策略：
  - weighted_average（默认，保守）：综合分数超阈值 OR 任一模型拦截 → 拦截
  - any_block：任一模型拦截 → 拦截
  - majority：多数模型投票决定

判定结果（action）：
  - 'block'  — 拦截
  - 'review' — 人工复核
  - 'pass'   — 放行
  - status='error' 时表示所有模型均检测失败
"""

import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FusionDetector:
    """多模型融合决策引擎"""

    def __init__(self, nsfw_detector, config: Dict = None):
        """
        初始化融合检测器

        Args:
            nsfw_detector: NSFWDetector 门面实例（用于调用各模型的 detect()）
            config:        全局配置字典，从中读取 nsfw_detection.fusion 配置
        """
        self.nsfw_detector = nsfw_detector

        # 各模型权重（权重越高，该模型在融合中的影响越大）
        self.weights = {
            'opennsfw2': 0.25,   # OpenNSFW2 权重
            'mobilenet': 0.30,   # MobileNet V2 140 权重
            'falconsai': 0.45,   # Falconsai ViT 权重（精度最高，权重最大）
        }
        # 融合决策阈值
        self.thresholds = {
            'block': 0.7,   # 综合分数 >= 此值 → 拦截
            'review': 0.4,  # 综合分数 >= 此值 → 复审
        }
        # 决策策略
        self.strategy = 'weighted_average'

        # 合法策略列表
        _valid_strategies = {'weighted_average', 'any_block', 'majority'}

        # 从配置文件覆盖默认值
        if config and 'nsfw_detection' in config:
            fusion = config['nsfw_detection'].get('fusion', {})
            for k, v in fusion.get('weights', {}).items():
                if k in self.weights:
                    self.weights[k] = float(v)
            for k, v in fusion.get('thresholds', {}).items():
                if k in self.thresholds:
                    self.thresholds[k] = float(v)
            if 'strategy' in fusion:
                s = fusion['strategy']
                if s in _valid_strategies:
                    self.strategy = s
                else:
                    logger.warning("FusionDetector: 无效策略 '%s', 使用默认 weighted_average", s)

        logger.info("FusionDetector 初始化完成, strategy=%s", self.strategy)

    def detect(self, image_path: str, models: Optional[List[str]] = None,
               thresholds: Optional[Dict[str, Dict]] = None) -> Dict:
        """
        融合多模型检测结果

        Args:
            image_path: 图片文件绝对路径
            models:     参与融合的模型 ID 列表，默认全部 ['opennsfw2', 'mobilenet', 'falconsai']
            thresholds: 按模型 ID 分发的阈值字典，如 {'mobilenet': {...}, 'opennsfw2': {...}}

        Returns:
            dict: 融合检测结果
                成功: {status:'success', fusion{final_score, final_porn, final_sexy, action, ...},
                       safety{色情,正常[,性感]}, content_type, model_results, elapsed_seconds}
                失败: {status:'error', message, model_results, elapsed_seconds}
        """
        start = time.perf_counter()

        if not models:
            models = ['opennsfw2', 'mobilenet', 'falconsai']

        logger.info("融合检测: 开始, models=%s, strategy=%s, image=%s",
                     models, self.strategy, image_path)

        # ---- 逐模型调用检测 ----
        model_results = {}       # 各模型的完整检测结果
        safety_scores = {}       # 各模型的色情分数（用于详情输出）
        per_model_ms = {}        # 各模型的耗时（ms），用于日志和返回
        weight_sum = 0.0         # 成功模型的权重总和（用于色情分数归一化）
        weighted_porn_sum = 0.0  # 色情分数加权和
        weighted_sexy_sum = 0.0  # 性感分数加权和（仅来自支持性感分类的模型）
        sexy_weight_sum = 0.0    # 提供性感分数的模型权重总和（独立于 weight_sum）
        fused_content_type = None  # 取第一个有内容分类的结果（仅 MobileNet 支持）

        for model_id in models:
            # 从 thresholds 中取出该模型对应的阈值（如果调用方传入了）
            model_th = thresholds.get(model_id) if thresholds else None
            model_start = time.perf_counter()
            result = self.nsfw_detector.detect(
                image_path, model_id=model_id, thresholds=model_th
            )
            # 外层兜底耗时：若模型未返回 elapsed_ms（如初始化失败），用外层差值补齐
            outer_ms = int(round((time.perf_counter() - model_start) * 1000))
            if 'elapsed_ms' not in result:
                result['elapsed_ms'] = outer_ms
            per_model_ms[model_id] = result.get('elapsed_ms', outer_ms)
            model_results[model_id] = result

            # 跳过检测失败的模型（不参与融合计算）
            if result.get('status') != 'success':
                logger.warning("融合检测: 模型 %s 检测失败, 跳过, elapsed=%dms",
                               model_id, per_model_ms[model_id])
                continue

            # 提取该模型的安全分类分数
            safety = result.get('safety', {})
            porn_score = safety.get('色情', 0.0)
            safety_scores[model_id] = porn_score

            # 按模型权重累加色情分数（所有模型都支持）
            w = self.weights.get(model_id, 0.3)
            weighted_porn_sum += porn_score * w
            weight_sum += w

            # 仅当模型实际返回性感分数时才累加（二分类模型不返回性感字段）
            if '性感' in safety:
                weighted_sexy_sum += safety['性感'] * w
                sexy_weight_sum += w

            # 取第一个有内容分类的结果（MobileNet 5-class 独有）
            if result.get('content_type') is not None and fused_content_type is None:
                fused_content_type = result['content_type']

        # 所有模型均失败，无法融合
        if weight_sum == 0:
            logger.warning("融合检测: 所有模型均检测失败, 无法融合, per_model_ms=%s",
                           per_model_ms)
            elapsed_seconds_raw = time.perf_counter() - start
            return {
                'status': 'error',
                'message': '没有可用的模型结果',
                'model_results': model_results,
                'elapsed_seconds': round(elapsed_seconds_raw, 2),
                'elapsed_ms': int(round(elapsed_seconds_raw * 1000)),
                'per_model_ms': per_model_ms,
            }

        # ---- 计算融合分数 ----
        # 色情融合分数 = 加权平均（所有成功模型参与）
        final_porn = round(weighted_porn_sum / weight_sum, 4)

        # 性感融合分数 = 仅在有模型实际提供性感分数时才计算
        # 二分类模型无法区分性感，不返回性感字段，因此不参与性感融合
        has_sexy = sexy_weight_sum > 0
        final_sexy = round(weighted_sexy_sum / sexy_weight_sum, 4) if has_sexy else None

        # 综合不安全分数 = 色情 + 性感（性感参与阈值评判）
        sexy_value = final_sexy if final_sexy is not None else 0.0
        combined_score = round(final_porn + sexy_value, 4)

        # 融合安全分类输出（仅包含模型实际支持的分类，避免理解偏差）
        fused_safety = {
            '色情': final_porn,
            '正常': round(max(0, 1.0 - combined_score), 4),
        }
        if has_sexy:
            fused_safety['性感'] = final_sexy

        # ---- 决策策略 ----
        block_th = self.thresholds['block']
        review_th = self.thresholds['review']

        if self.strategy == 'any_block':
            # any_block 策略：任一模型拦截 → 拦截，否则用综合分数判定
            any_block = any(
                r.get('action') == 'block'
                for r in model_results.values()
                if r.get('status') == 'success'
            )
            if any_block:
                action = 'block'
            elif combined_score >= review_th:
                action = 'review'
            else:
                action = 'pass'

        elif self.strategy == 'majority':
            # majority 策略：多数模型投票决定
            actions = [
                r.get('action')
                for r in model_results.values()
                if r.get('status') == 'success'
            ]
            block_count = actions.count('block')
            review_count = actions.count('review')
            total = len(actions)
            if block_count > total / 2:
                action = 'block'
            elif (block_count + review_count) > total / 2:
                action = 'review'
            else:
                action = 'pass'

        else:
            # weighted_average 策略（默认，保守）：
            # 综合分数超阈值 OR 任一模型拦截 → 拦截
            any_block = any(
                r.get('action') == 'block'
                for r in model_results.values()
                if r.get('status') == 'success'
            )
            if any_block or combined_score >= block_th:
                action = 'block'
            elif combined_score >= review_th:
                action = 'review'
            else:
                action = 'pass'

        action_text = {'block': '拦截', 'review': '复审', 'pass': '放行'}

        # 构建详情说明
        details = [f"{mid}: 色情 {score:.2%}" for mid, score in safety_scores.items()]
        details.append(f"融合色情: {final_porn:.2%}")
        if has_sexy and final_sexy > 0:
            details.append(f"融合性感: {final_sexy:.2%}")
        details.append(f"综合分数: {combined_score:.2%}")

        elapsed_seconds_raw = time.perf_counter() - start
        elapsed = round(elapsed_seconds_raw, 2)
        elapsed_ms = int(round(elapsed_seconds_raw * 1000))
        logger.info("融合检测: 完成, action=%s, strategy=%s, combined_score=%.4f, "
                     "final_porn=%.4f, final_sexy=%s, model_scores=%s, "
                     "per_model_ms=%s, elapsed=%dms (%.2fs)",
                     action, self.strategy, combined_score,
                     final_porn, final_sexy, safety_scores,
                     per_model_ms, elapsed_ms, elapsed)

        return {
            'status': 'success',
            'fusion': {
                'final_score': combined_score,  # 综合不安全分数（色情 + 性感，若有）
                'final_porn': final_porn,       # 融合色情分数
                'final_sexy': final_sexy,       # 融合性感分数（无模型支持时为 None）
                'action': action,
                'action_text': action_text[action],
                'strategy': self.strategy,
                'model_scores': safety_scores,
                'details': details,
            },
            'content_type': fused_content_type,
            'safety': fused_safety,
            'model_results': model_results,
            'elapsed_seconds': elapsed,
            'elapsed_ms': elapsed_ms,
            'per_model_ms': per_model_ms,
        }
