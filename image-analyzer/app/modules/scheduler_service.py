"""
定时检测服务模块

定期从第三方接口拉取待检测数据（直播截图 URL 列表），逐条执行三维融合检测，
将结果回调通知第三方。同时暴露执行历史和状态，供可视化页面查询。

status 映射:
  0=已关播  1=待检测  2=检测中  3=通过  4=疑似  5=检测失败

riskLevel 映射（基于 fusion_score）:
  < dynamic_threshold (0.75)  → 正常 (status=3)
  [0.75, risk_mid)            → 疑似 (status=4)
  [risk_mid, static_threshold)→ 中风险挂播 (status=4)
  >= static_threshold (0.95)  → 高风险挂播 (status=4)
"""

import os
import sys
import uuid
import time
import threading
import traceback
import requests
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler


def _log(msg, *args):
    """打印日志到 stdout（gunicorn/Docker 下可靠输出）"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    text = msg % args if args else msg
    print(f"[{ts}] [scheduler] {text}", flush=True)


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
        self._last_run = None
        self._current_step = ''       # 当前执行步骤（供前端轮询）

        # 阈值
        thresholds = self.motion_config.get('thresholds', {})
        self.dynamic_threshold = float(thresholds.get('dynamic', 0.75))
        self.static_threshold = float(thresholds.get('static', 0.95))
        self.risk_mid = float(self.scheduler_config.get('risk_mid', 0.85))
        self.max_history = int(self.scheduler_config.get('max_history', 50))

        # HTTP 配置
        api_config = self.scheduler_config.get('api', {})
        self.base_url = api_config.get('base_url', '').rstrip('/')
        self.fetch_path = api_config.get('fetch_path', '/sano/dyCheck/data')
        self.callback_path = api_config.get('callback_path', '/sano/dyCheck/call/check')
        self.timeout = int(self.scheduler_config.get('timeout', 30))
        self.image_download_timeout = int(self.scheduler_config.get('image_download_timeout', 15))

        _log("SchedulerService 初始化完成, base_url=%s, fetch=%s, callback=%s",
             self.base_url, self.fetch_path, self.callback_path)

    def start(self):
        """启动定时调度器"""
        interval = int(self.scheduler_config.get('interval_minutes', 10))
        self.scheduler.add_job(
            self.run_batch, 'interval', minutes=interval,
            id='scheduler_batch', replace_existing=True,
        )
        self.scheduler.start()
        _log("定时检测服务已启动，间隔 %d 分钟", interval)

    def stop(self):
        """停止调度器"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            _log("定时检测服务已停止")

    def run_batch(self):
        """单次批次执行（定时触发或手动触发）"""
        if self._running:
            _log("上一批次仍在执行，跳过本次触发")
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
            'steps': [],       # 记录每一步的日志
        }

        def step(msg):
            ts = datetime.now().strftime('%H:%M:%S')
            batch_record['steps'].append(f"[{ts}] {msg}")
            self._current_step = msg
            _log(msg)

        try:
            # 1. 拉取待检测数据
            fetch_url = self.base_url + self.fetch_path
            step(f"正在拉取数据: GET {fetch_url}")
            resp = requests.get(fetch_url, timeout=self.timeout)
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
            _log("异常堆栈:\n%s", traceback.format_exc())
            batch_record['error'] = err_msg
        finally:
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
            'riskLevel': None,      # 英文 code: low/medium/high/critical
            'riskLabel': None,      # 中文标签: 正常/疑似/中风险挂播/高风险挂播
            'scores': {},
            'callback_ok': False,
            'callback_error': None,
            'error': None,
            'elapsed': 0,           # 耗时（秒）
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
            for i, url in enumerate(pic_urls):
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
            step(f"  item {item_id}: 检测完成, status={detect_result.get('status')}, result={detect_result.get('result')}")

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

            status, risk_code, risk_label = self._map_risk(fusion_score, result_label)
            record['status'] = status
            record['riskLevel'] = risk_code
            record['riskLabel'] = risk_label
            step(f"  item {item_id}: fusion={fusion_score:.4f}, status={status}, risk={risk_code}({risk_label})")

            # 回调
            self._do_callback(record, step)

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            step(f"  item {item_id}: 处理异常 - {err_msg}")
            _log("item %s 异常堆栈:\n%s", item_id, traceback.format_exc())
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

    def _download_image(self, url, step):
        """下载单张图片，返回本地路径"""
        try:
            resp = requests.get(url, timeout=self.image_download_timeout)
            resp.raise_for_status()

            # 从 URL 提取文件名
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
        if fusion_score < self.dynamic_threshold:
            return 3, 'low', '正常'
        elif fusion_score < self.risk_mid:
            return 4, 'medium', '疑似'
        elif fusion_score < self.static_threshold:
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
            'riskLevel': record['riskLevel'],       # 英文 code: low/medium/high/critical
            'checkEndTime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        step(f"  回调: POST {callback_url}, payload={payload}")
        try:
            resp = requests.post(callback_url, json=payload, timeout=self.timeout)
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
            'risk_mid': self.risk_mid,
            'dynamic_threshold': self.dynamic_threshold,
            'static_threshold': self.static_threshold,
            'max_history': self.max_history,
        }

    def trigger_manual(self):
        """手动触发一次批次执行（子线程运行，不阻塞 Flask）"""
        if self._running:
            return False, '上一批次仍在执行中'
        t = threading.Thread(target=self.run_batch, daemon=True)
        t.start()
        return True, '已触发执行'
