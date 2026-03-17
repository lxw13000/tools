"""
统一日志配置模块

为整个应用提供统一的日志配置，同时输出到：
  1. stdout — 供 Docker logs / gunicorn 捕获
  2. RotatingFileHandler — 落盘到 .log 文件，自动轮转

生产环境要求：
  - 日志包含线程信息（多线程推理环境必须）
  - 第三方库仅输出 WARNING 以上（避免刷屏）
  - 定时服务日志独立文件且不重复输出到主日志
  - 文件自动轮转（RotatingFileHandler），防止磁盘占满

调用方式：
  在 app.py 最早期调用 setup_logging(config)，
  后续所有模块 logging.getLogger(__name__) 自动继承配置。
"""

import os
import logging
from logging.handlers import RotatingFileHandler

# 默认值
DEFAULT_LOG_DIR = '/app/logs'
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_MAX_BYTES = 50 * 1024 * 1024   # 50MB
DEFAULT_BACKUP_COUNT = 5               # 保留 5 个历史文件
DEFAULT_LOG_FORMAT = (
    '[%(asctime)s] [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s'
)
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(config: dict = None):
    """
    初始化全局日志配置

    Args:
        config: 完整的 config.yaml 字典，从中读取 logging 子项
    """
    config = config or {}
    log_config = config.get('logging', {})

    log_dir = log_config.get('log_dir', DEFAULT_LOG_DIR)
    log_level_str = log_config.get('level', DEFAULT_LOG_LEVEL).upper()
    max_bytes = int(log_config.get('max_bytes', DEFAULT_MAX_BYTES))
    backup_count = int(log_config.get('backup_count', DEFAULT_BACKUP_COUNT))

    log_level = getattr(logging, log_level_str, logging.INFO)

    # 日志格式（包含线程名，便于多线程环境定位问题）
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)

    # 根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有 handler（防止重复初始化）
    root_logger.handlers.clear()

    # Handler 1: stdout（Docker logs 可见）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Handler 2: 应用日志文件（RotatingFileHandler，自动轮转）
    os.makedirs(log_dir, exist_ok=True)
    app_log_path = os.path.join(log_dir, 'app.log')
    file_handler = RotatingFileHandler(
        app_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8',
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Handler 3: 定时检测专用日志文件（独立输出，不重复写入主日志）
    scheduler_log_path = os.path.join(log_dir, 'scheduler.log')
    scheduler_handler = RotatingFileHandler(
        scheduler_log_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8',
    )
    scheduler_handler.setLevel(log_level)
    scheduler_handler.setFormatter(formatter)
    scheduler_logger = logging.getLogger('app.modules.scheduler_service')
    scheduler_logger.addHandler(scheduler_handler)
    scheduler_logger.propagate = False  # 阻止向根 logger 冒泡，避免重复输出

    # Handler 4: gunicorn access/error 日志轮转
    # gunicorn 启动时通过 --access-logfile / --error-logfile 写入文件，
    # 但其内置 handler 无轮转机制，这里为 gunicorn logger 加挂轮转 handler
    for glog_name in ['gunicorn.access', 'gunicorn.error']:
        glog = logging.getLogger(glog_name)
        glog_path = os.path.join(log_dir, f'{glog_name.split(".")[-1]}.log')
        glog_handler = RotatingFileHandler(
            glog_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8',
        )
        glog_handler.setLevel(log_level)
        glog_handler.setFormatter(formatter)
        glog.addHandler(glog_handler)

    # 降低第三方库的日志级别，避免刷屏
    for noisy in ['urllib3', 'PIL', 'tensorflow', 'h5py', 'apscheduler',
                   'werkzeug', 'absl', 'transformers']:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root_logger.info("日志系统初始化完成: level=%s, dir=%s, max=%dMB x %d files",
                     log_level_str, log_dir, max_bytes // (1024 * 1024), backup_count)
