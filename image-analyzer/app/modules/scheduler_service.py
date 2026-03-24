"""
定时检测服务模块

定期从第三方接口拉取待检测数据（直播截图 URL 列表），逐条执行三维融合检测，
将结果回调通知第三方。同时暴露执行历史和状态，供可视化页面查询。

status 映射:
  0=已关播  1=待检测  2=检测中  3=通过  4=疑似  5=检测失败

riskLevel 映射（基于 fusion_score 与三阈值）:
  < review_threshold (0.78)    → low (status=3)
  [review, mid_risk)           → medium (status=4)
  [mid_risk, high_risk)        → high (status=4)
  >= high_risk_threshold (0.95)→ critical (status=4)
"""

import os
import uuid
import time
import logging
import threading
import traceback
import requests
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)


class SchedulerService:
    """定时检测服务：拉取、检测、回调"""

    def __init__(self, motion_detector, config):
        self.motion_detector = motion_detector
        self.config = config or {}
        self.scheduler_config = self.config.get('scheduler', {})
        self.motion_config = self.config.get('motion_detection', {})

        self.scheduler = BackgroundScheduler()
        self.history = []
        self._running = False
        self._lock = threading.Lock()       # 保护 _running 的互斥锁
        self._last_run = None
        self._current_step = ''

        # 阈值（三阈值四分类，与 motion_detector 一致）
        thresholds = self.motion_config.get('thresholds', {})
        self.review_threshold = float(thresholds.get('review', 0.78))
        self.mid_risk_threshold = float(thresholds.get('mid_risk', 0.87))
        self.high_risk_threshold = float(thresholds.get('high_risk', 0.95))
        self.max_history = int(self.scheduler_config.get('max_history', 50))

        # HTTP 配置
        api_config = self.scheduler_config.get('api', {})
        self.base_url = api_config.get('base_url', '').rstrip('/')
        self.fetch_path = api_config.get('fetch_path', '/sano/dyCheck/data')
        self.callback_path = api_config.get('callback_path', '/sano/dyCheck/call/check')
        self.timeout = int(self.scheduler_config.get('timeout', 30))
        self.image_download_timeout = int(self.scheduler_config.get('image_download_timeout', 15))

        logger.info("SchedulerService 初始化完成, base_url=%s, fetch=%s, callback=%s",
                     self.base_url, self.fetch_path, self.callback_path)

    def start(self):
        """启动定时调度器"""
        interval = int(self.scheduler_config.get('interval_minutes', 10))
        self.scheduler.add_job(
            self.run_batch, 'interval', minutes=interval,
            id='scheduler_batch', replace_existing=True,
        )
        self.scheduler.start()
        logger.info("定时检测服务已启动，间隔 %d 分钟", interval)

    def stop(self):
        """停止调度器"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("定时检测服务已停止")

    def run_batch(self):
        """单次批次执行（定时触发或手动触发）"""
        # 用锁保护 _running 标志，防止竞态
        with self._lock:
            if self._running:
                logger.warning("上一批次仍在执行，跳过本次触发")
                return
            self._running = True

        self._current_step = '开始执行'
        batch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._last_run = batch_time
        batch_record = {
            'time': batch_time,
            'fetch_count': 0,
            'items': [],
            'error': None,
            'steps': [],
        }

        def step(msg):
            ts = datetime.now().strftime('%H:%M:%S')
            batch_record['steps'].append(f"[{ts}] {msg}")
            self._current_step = msg
            logger.info("[batch] %s", msg)

        try:
            # 1. 拉取待检测数据
            fetch_url = self.base_url + self.fetch_path
            step(f"正在拉取数据: GET {fetch_url}")
            resp = requests.get(fetch_url, timeout=self.timeout, headers=self._HEADERS)
            step(f"拉取响应: HTTP {resp.status_code}, 长度 {len(resp.content)} bytes")
            resp.raise_for_status()

            payload = resp.json()
            step(f"响应JSON解析成功, 顶层 keys: {list(payload.keys())}")

            data_list = payload.get('data', [])
            if not isinstance(data_list, list):
                step(f"data 字段不是数组, 类型={type(data_list).__name__}, 置为空列表")
                data_list = []

            batch_record['fetch_count'] = len(data_list)
            step(f"拉取到 {len(data_list)} 条待检测数据")

            # 2. 逐条处理
            for idx, item in enumerate(data_list):
                step(f"处理第 {idx+1}/{len(data_list)} 条, id={item.get('id', '?')}")
                result = self._process_item(item, step)
                batch_record['items'].append(result)

            step(f"批次执行完成, 共处理 {len(batch_record['items'])} 条")

        except Exception as e:
            err_msg = f"批次执行异常: {type(e).__name__}: {e}"
            step(err_msg)
            logger.error("批次异常堆栈:\n%s", traceback.format_exc())
            batch_record['error'] = err_msg
        finally:
            with self._lock:
                self._running = False
            self._current_step = ''
            # FIFO 保留最近 max_history 条
            self.history.insert(0, batch_record)
            if len(self.history) > self.max_history:
                self.history = self.history[:self.max_history]

    def _process_item(self, item, step):
        """处理单条检测任务"""
        item_id = item.get('id', 'unknown')
        pic_urls = item.get('picUrls', [])
        item_start = time.time()
        record = {
            'id': item_id,
            'picUrls': pic_urls,
            'status': None,
            'riskLevel': None,
            'riskLabel': None,
            'scores': {},
            'callback_ok': False,
            'callback_error': None,
            'error': None,
            'elapsed': 0,
        }
        temp_files = []

        try:
            if not pic_urls or len(pic_urls) < 2:
                record['status'] = 5
                record['riskLevel'] = 'error'
                record['riskLabel'] = '检测失败'
                record['error'] = f'图片数量不足（需要>=2，实际={len(pic_urls) if pic_urls else 0}）'
                step(f"  item {item_id}: 图片不足, 跳过检测")
                self._do_callback(record, step)
                return record

            # 下载图片到临时目录
            step(f"  item {item_id}: 开始下载 {len(pic_urls)} 张图片")
            for url in pic_urls:
                local_path = self._download_image(url, step)
                if local_path:
                    temp_files.append(local_path)
            step(f"  item {item_id}: 下载完成, 有效 {len(temp_files)}/{len(pic_urls)} 张")

            if len(temp_files) < 2:
                record['status'] = 5
                record['riskLevel'] = 'error'
                record['riskLabel'] = '检测失败'
                record['error'] = f'有效图片下载不足 2 张（成功 {len(temp_files)} 张）'
                self._do_callback(record, step)
                return record

            # 调用动态检测器
            step(f"  item {item_id}: 开始三维融合检测...")
            detect_result = self.motion_detector.detect(temp_files)
            step(f"  item {item_id}: 检测完成, status={detect_result.get('status')}, "
                 f"result={detect_result.get('result')}")

            if detect_result.get('status') == 'error':
                record['status'] = 5
                record['riskLevel'] = 'error'
                record['riskLabel'] = '检测失败'
                record['error'] = detect_result.get('message', '检测异常')
                self._do_callback(record, step)
                return record

            fusion_score = detect_result.get('fusion_score', 0)
            result_label = detect_result.get('result', 'review')
            scores = detect_result.get('scores', {})

            record['scores'] = {
                'zombieScore': round(fusion_score, 4),
                'ssimScore': round(scores.get('ssim', 0), 4),
                'flowScore': round(scores.get('flow', 0), 4),
                'phashScore': round(scores.get('phash', 0), 4),
            }

            # 附加人脸检测分数（仅在启用时存在）
            if detect_result.get('face_detection_used'):
                record['scores']['originalZombieScore'] = round(
                    detect_result.get('original_fusion_score', fusion_score), 4)
                if detect_result.get('composite_detected'):
                    record['scores']['faceStaticScore'] = round(
                        detect_result.get('face_static_score', 0), 4)
                    record['scores']['compositeDetected'] = True
                else:
                    record['scores']['faceScore'] = round(
                        detect_result.get('face_change_score', 0), 4)

            status, risk_code, risk_label = self._map_risk(fusion_score, result_label)
            record['status'] = status
            record['riskLevel'] = risk_code
            record['riskLabel'] = risk_label
            step(f"  item {item_id}: fusion={fusion_score:.4f}, "
                 f"status={status}, risk={risk_code}({risk_label})")

            # 回调
            self._do_callback(record, step)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            step(f"  item {item_id}: 处理异常 - {err_msg}")
            logger.error("item %s 异常堆栈:\n%s", item_id, traceback.format_exc())
            record['status'] = 5
            record['riskLevel'] = 'error'
            record['riskLabel'] = '检测失败'
            record['error'] = err_msg
            try:
                self._do_callback(record, step)
            except Exception:
                pass
        finally:
            record['elapsed'] = round(time.time() - item_start, 2)
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

        return record

    # 浏览器 UA，避免被 CDN/对象存储拦截
    _HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0.0.0 Safari/537.36',
    }

    def _download_image(self, url, step):
        """下载单张图片，返回本地路径"""
        try:
            resp = requests.get(
                url,
                timeout=(self.image_download_timeout, self.image_download_timeout * 2),
                headers=self._HEADERS,
            )
            resp.raise_for_status()

            # 限制最大 50MB，防止内存溢出
            if len(resp.content) > 50 * 1024 * 1024:
                step(f"    图片过大 ({len(resp.content)} bytes)，跳过: {url}")
                return None

            filename = url.split('/')[-1].split('?')[0] or 'image.jpg'
            local_name = f"scheduler_{uuid.uuid4().hex[:8]}_{filename}"
            upload_dir = '/tmp/uploads'
            os.makedirs(upload_dir, exist_ok=True)
            local_path = os.path.join(upload_dir, local_name)

            with open(local_path, 'wb') as f:
                f.write(resp.content)

            step(f"    下载成功: {filename} ({len(resp.content)} bytes)")
            return local_path
        except Exception as e:
            step(f"    下载失败: {url} - {type(e).__name__}: {e}")
            return None

    def _map_risk(self, fusion_score, result):
        """映射 status、riskLevel code 和中文标签"""
        if result == 'error':
            return 5, 'error', '检测失败'
        if fusion_score < self.review_threshold:
            return 3, 'low', '正常'
        elif fusion_score < self.mid_risk_threshold:
            return 4, 'medium', '人工复核'
        elif fusion_score < self.high_risk_threshold:
            return 4, 'high', '中风险挂播'
        else:
            return 4, 'critical', '高风险挂播'

    def _do_callback(self, record, step):
        """POST 回调检测结果到第三方"""
        callback_url = self.base_url + self.callback_path
        payload = {
            'id': record['id'],
            'status': record['status'],
            'zombieScore': record.get('scores', {}).get('zombieScore', 0),
            'ssimScore': record.get('scores', {}).get('ssimScore', 0),
            'flowScore': record.get('scores', {}).get('flowScore', 0),
            'phashScore': record.get('scores', {}).get('phashScore', 0),
            'riskLevel': record['riskLevel'],
            'checkEndTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 人脸检测分数（仅在启用时附加）
        scores = record.get('scores', {})
        if 'originalZombieScore' in scores:
            payload['originalZombieScore'] = scores['originalZombieScore']
        if 'faceScore' in scores:
            payload['faceScore'] = scores['faceScore']
        if scores.get('compositeDetected'):
            payload['faceStaticScore'] = scores.get('faceStaticScore', 0)
            payload['compositeDetected'] = True

        step(f"  回调: POST {callback_url}, payload={payload}")
        try:
            resp = requests.post(callback_url, json=payload, timeout=self.timeout,
                                 headers=self._HEADERS)
            step(f"  回调响应: HTTP {resp.status_code}, body={resp.text[:200]}")
            resp.raise_for_status()
            record['callback_ok'] = True
        except Exception as e:
            record['callback_ok'] = False
            record['callback_error'] = f"{type(e).__name__}: {e}"
            step(f"  回调失败: {record['callback_error']}")

    def get_status(self):
        """返回服务状态（供 API 调用）"""
        next_run = None
        job = self.scheduler.get_job('scheduler_batch')
        if job and job.next_run_time:
            next_run = job.next_run_time.strftime('%Y-%m-%d %H:%M:%S')

        return {
            'enabled': self.scheduler_config.get('enabled', False),
            'running': self._running,
            'current_step': self._current_step,
            'interval_minutes': int(self.scheduler_config.get('interval_minutes', 10)),
            'last_run': self._last_run,
            'next_run': next_run,
            'history': self.history,
        }

    def get_config_info(self):
        """返回服务配置（供 API 调用）"""
        return {
            'enabled': self.scheduler_config.get('enabled', False),
            'interval_minutes': int(self.scheduler_config.get('interval_minutes', 10)),
            'base_url': self.base_url,
            'fetch_path': self.fetch_path,
            'callback_path': self.callback_path,
            'timeout': self.timeout,
            'image_download_timeout': self.image_download_timeout,
            'review_threshold': self.review_threshold,
            'mid_risk_threshold': self.mid_risk_threshold,
            'high_risk_threshold': self.high_risk_threshold,
            'max_history': self.max_history,
        }

    def trigger_manual(self):
        """手动触发一次批次执行（子线程运行，不阻塞 Flask）"""
        with self._lock:
            if self._running:
                return False, '上一批次仍在执行中'
        t = threading.Thread(target=self.run_batch, daemon=True)
        t.start()
        return True, '已触发执行'
