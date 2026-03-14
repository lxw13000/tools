"""
多模型融合决策引擎
通过 NSFWDetector 门面调用多个模型，加权融合安全分类分数
"""

import time
from typing import Dict, List, Optional


class FusionDetector:

    def __init__(self, nsfw_detector, config: Dict = None):
        self.nsfw_detector = nsfw_detector

        self.weights = {
            'opennsfw2': 0.25,
            'mobilenet': 0.30,
            'falconsai': 0.45,
        }
        self.thresholds = {
            'block': 0.7,
            'review': 0.4,
        }
        self.strategy = 'weighted_average'

        if config and 'nsfw_detection' in config:
            fusion = config['nsfw_detection'].get('fusion', {})
            for k, v in fusion.get('weights', {}).items():
                if k in self.weights:
                    self.weights[k] = float(v)
            for k, v in fusion.get('thresholds', {}).items():
                if k in self.thresholds:
                    self.thresholds[k] = float(v)
            if 'strategy' in fusion:
                self.strategy = fusion['strategy']

    def detect(self, image_path: str, models: Optional[List[str]] = None) -> Dict:
        start = time.time()

        if not models:
            models = ['opennsfw2', 'mobilenet', 'falconsai']

        model_results = {}
        safety_scores = {}
        weight_sum = 0.0
        weighted_sum = 0.0
        fused_content_type = None

        for model_id in models:
            result = self.nsfw_detector.detect(image_path, model_id=model_id)
            model_results[model_id] = result

            if result.get('status') != 'success':
                continue

            nsfw_score = result.get('safety', {}).get('色情', 0.0)
            safety_scores[model_id] = nsfw_score

            w = self.weights.get(model_id, 0.3)
            weighted_sum += nsfw_score * w
            weight_sum += w

            # 取第一个有内容分类的结果（仅 MobileNet）
            if result.get('content_type') is not None and fused_content_type is None:
                fused_content_type = result['content_type']

        if weight_sum == 0:
            return {
                'status': 'error',
                'message': '没有可用的模型结果',
                'model_results': model_results,
                'elapsed_seconds': round(time.time() - start, 2),
            }

        final_score = round(weighted_sum / weight_sum, 4)

        fused_safety = {
            '色情': final_score,
            '暴力': 0.0,
            '正常': round(1.0 - final_score, 4),
        }

        # 决策
        block_th = self.thresholds['block']
        review_th = self.thresholds['review']

        if self.strategy == 'any_block':
            any_block = any(
                r.get('action') == 'block'
                for r in model_results.values()
                if r.get('status') == 'success'
            )
            if any_block:
                action = 'block'
            elif final_score >= review_th:
                action = 'review'
            else:
                action = 'pass'
        elif self.strategy == 'majority':
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
            # weighted_average（默认）+ 保守策略
            any_block = any(
                r.get('action') == 'block'
                for r in model_results.values()
                if r.get('status') == 'success'
            )
            if any_block or final_score >= block_th:
                action = 'block'
            elif final_score >= review_th:
                action = 'review'
            else:
                action = 'pass'

        action_text = {'block': '拦截', 'review': '复审', 'pass': '放行'}

        details = [f"{mid}: {score:.2%}" for mid, score in safety_scores.items()]
        details.append(f"融合分数: {final_score:.2%}")

        return {
            'status': 'success',
            'fusion': {
                'final_score': final_score,
                'action': action,
                'action_text': action_text[action],
                'strategy': self.strategy,
                'model_scores': safety_scores,
                'details': details,
            },
            'content_type': fused_content_type,
            'safety': fused_safety,
            'model_results': model_results,
            'elapsed_seconds': round(time.time() - start, 2),
        }
