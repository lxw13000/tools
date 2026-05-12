"""
Flask API 服务主文件

提供图片分析的 REST API 接口，包含两大核心功能：
  1. 图片动态检测 — POST /api/detect/motion（1-6 张图片，判断静态/动态）
  2. 图片内容安全检测（鉴黄）— POST /api/detect/nsfw（1 张图片，判断安全分类）
  3. 多模型融合检测 — POST /api/detect/nsfw/fusion（1 张图片，多模型综合评分）

判定结果类型：放行(pass) / 拦截(block) / 人工复核(review) / 失败(error)
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import uuid
import logging
import yaml
from .logging_config import setup_logging
from .modules import MotionDetector, NSFWDetector, FusionDetector, SchedulerService, NsfwService

# 动态检测允许的最大图片数量
MAX_MOTION_IMAGES = 6

# ---- Flask 应用初始化 ----
app = Flask(__name__)
CORS(app)


def load_config():
    """加载 config.yaml 配置文件，失败时返回 None（使用内置默认值）"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        # 此时 logger 尚未初始化，用 print 输出到 stderr
        import sys
        print(f"[WARN] 无法加载配置文件 {config_path}, 使用默认配置. 错误: {e}", file=sys.stderr)
        return None


config = load_config()

# ---- 初始化日志系统（最早执行，后续所有模块继承配置）----
setup_logging(config)
logger = logging.getLogger(__name__)

# ---- 应用配置 ----
if config:
    server_config = config.get('server', {})
    upload_config = config.get('upload', {})
    app.config['MAX_CONTENT_LENGTH'] = server_config.get('max_content_length', 52428800)
    app.config['UPLOAD_FOLDER'] = upload_config.get('folder', '/tmp/uploads')
    app.config['ALLOWED_EXTENSIONS'] = set(upload_config.get('allowed_extensions',
                                           ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']))
else:
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---- 初始化检测器 ----
motion_config = config.get('motion_detection', {}) if config else {}
motion_detector = MotionDetector(config=motion_config)
nsfw_detector = NSFWDetector(
    models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
    config=config,
)
fusion_detector = FusionDetector(nsfw_detector=nsfw_detector, config=config)

# ---- 初始化定时检测服务 ----
scheduler_service = SchedulerService(motion_detector=motion_detector, config=config)
scheduler_config = config.get('scheduler', {}) if config else {}
if scheduler_config.get('enabled', False):
    scheduler_service.start()

# ---- 初始化 NSFW 检测服务（供外部业务调用）----
nsfw_service = NsfwService(
    nsfw_detector=nsfw_detector,
    fusion_detector=fusion_detector,
    config=config,
)


def allowed_file(filename: str) -> bool:
    """检查上传文件扩展名是否在允许列表中"""
    if not filename:
        return False
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ---- 页面路由 ----

@app.route('/')
def index():
    """首页导航"""
    return render_template('index.html')


@app.route('/motion')
def motion_page():
    """动态检测页面"""
    return render_template('motion.html')


@app.route('/nsfw')
def nsfw_page():
    """内容安全检测页面"""
    return render_template('nsfw.html')


@app.route('/scheduler')
def scheduler_page():
    """定时检测服务页面"""
    return render_template('scheduler.html')


@app.route('/nsfw-service')
def nsfw_service_page():
    """NSFW 检测服务测试页面"""
    return render_template('nsfw_service.html')


# ---- API 路由 ----

@app.route('/api/health', methods=['GET'])
def health_check():
    """GET /api/health — 健康检查"""
    return jsonify({"status": "ok", "message": "服务运行正常"})


@app.route('/api/nsfw/models', methods=['GET'])
def get_nsfw_models():
    """GET /api/nsfw/models — 返回所有注册模型及可用性"""
    return jsonify(nsfw_detector.get_models_info())


@app.route('/api/nsfw/config', methods=['GET'])
def get_nsfw_config():
    """
    GET /api/nsfw/config — 返回所有模型的默认阈值（供前端渲染滑块初始值）

    返回结构（与 config.yaml 中的 nsfw_detection 一致）：
        {
            "mobilenet": {porn, hentai, sexy, porn_hentai},
            "opennsfw2": {nsfw_block, nsfw_review},
            "falconsai": {nsfw_block, nsfw_review},
            "fusion":    {weights{...}, thresholds{block, review}, strategy},
            # 顶层兼容字段（仅 mobilenet 阈值，旧前端可直接取）
            "porn": ..., "hentai": ..., "sexy": ..., "porn_hentai": ...,
        }
    """
    all_defaults = nsfw_detector.get_all_default_thresholds()
    # 兼容旧前端：顶层平铺 mobilenet 阈值
    result = {**all_defaults['mobilenet'], **all_defaults}
    return jsonify(result)


@app.route('/api/motion/config', methods=['GET'])
def get_motion_config():
    """GET /api/motion/config — 返回动态检测默认权重和阈值（供前端渲染滑块初始值）"""
    return jsonify({
        "weights": motion_detector.weights,
        "thresholds": motion_detector.thresholds,
        "face_detection": {
            "enabled": motion_detector.face_detector.enabled,
            "available": motion_detector.face_detector.is_available(),
        },
    })


@app.route('/api/detect/motion', methods=['POST'])
def detect_motion():
    """
    POST /api/detect/motion — 图片序列动态检测

    参数（multipart form）：
      - images: 多文件字段，1-6 张图片

    返回：
      - 成功: {status:'success', result:'static'|'dynamic', confidence, ...}
      - 失败: {status:'error', message}
    """
    if 'images' not in request.files:
        return jsonify({"status": "error", "message": "请上传图片文件"}), 400

    files = request.files.getlist('images')
    if not files or len(files) == 0:
        return jsonify({"status": "error", "message": "请至少上传一张图片"}), 400

    # 校验图片数量上限（业务规则：最多 6 张）
    if len(files) > MAX_MOTION_IMAGES:
        return jsonify({
            "status": "error",
            "message": f"图片数量超出限制，最多支持 {MAX_MOTION_IMAGES} 张",
        }), 400

    saved_paths = []
    session_id = str(uuid.uuid4())  # 用于临时文件名去重

    try:
        # 保存上传文件到临时目录
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{session_id}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                saved_paths.append(filepath)

        if not saved_paths:
            return jsonify({"status": "error", "message": "没有有效的图片文件"}), 400

        # 解析前端传来的可选权重参数
        weights = {}
        for key in ['phash', 'ssim', 'flow']:
            val = request.form.get(f'weight_{key}', type=float)
            if val is not None:
                weights[key] = val

        # 解析前端传来的可选阈值参数
        thresholds = {}
        for key in ['high_risk', 'mid_risk', 'review']:
            val = request.form.get(f'threshold_{key}', type=float)
            if val is not None:
                thresholds[key] = val

        # 解析前端传来的可选人脸检测开关（per-request 覆盖）
        face_override = None
        face_param = request.form.get('face_detection')
        if face_param is not None:
            face_override = face_param.lower() in ('true', '1', 'yes')

        # 调用动态检测器
        return jsonify(motion_detector.detect(
            saved_paths,
            weights=weights if weights else None,
            thresholds=thresholds if thresholds else None,
            face_detection_enabled=face_override,
        ))

    except Exception as e:
        logger.exception("动态检测接口异常")
        return jsonify({"status": "error", "message": f"处理失败: {str(e)}"}), 500

    finally:
        # 无论成功失败，清理所有临时文件
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as e:
                logger.warning("临时文件清理失败 %s: %s", path, e)


@app.route('/api/detect/nsfw', methods=['POST'])
def detect_nsfw():
    """
    POST /api/detect/nsfw — 单模型 NSFW 检测

    参数（multipart form）：
      - image:    单文件字段，1 张图片
      - model_id: 模型标识（mobilenet / opennsfw2 / falconsai），默认 mobilenet
      - threshold_*: 阈值参数（根据模型类型不同，key 不同）

    返回：
      - 成功: {status:'success', action:'pass'|'block'|'review', safety, ...}
      - 失败: {status:'error', message}
    """
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "请上传图片文件"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "不支持的文件格式"}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath)

        model_id = request.form.get('model_id', 'mobilenet')

        # 根据模型类型解析前端传来的阈值参数
        thresholds = {}
        if model_id == 'mobilenet':
            # MobileNet 5-class 阈值: porn, hentai, sexy, porn_hentai
            for key in ['porn', 'hentai', 'sexy', 'porn_hentai']:
                val = request.form.get(f'threshold_{key}', type=float)
                if val is not None:
                    thresholds[key] = val
        else:
            # 二分类模型阈值: nsfw_block, nsfw_review
            for key in ['nsfw_block', 'nsfw_review']:
                val = request.form.get(f'threshold_{key}', type=float)
                if val is not None:
                    thresholds[key] = val

        result = nsfw_detector.detect(
            filepath,
            model_id=model_id,
            thresholds=thresholds if thresholds else None,
        )
        return jsonify(result)

    except Exception as e:
        logger.exception("NSFW 检测接口异常")
        return jsonify({"status": "error", "message": f"处理失败: {str(e)}"}), 500

    finally:
        # 清理临时文件
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError as e:
            logger.warning("临时文件清理失败 %s: %s", filepath, e)


@app.route('/api/detect/nsfw/fusion', methods=['POST'])
def detect_nsfw_fusion():
    """
    POST /api/detect/nsfw/fusion — 多模型融合检测

    参数（multipart form）：
      - image:    单文件字段，1 张图片
      - models:   逗号分隔的模型 ID 列表（如 "opennsfw2,mobilenet,falconsai"）
      - threshold_*: 阈值参数（同时传 MobileNet 和二分类模型的阈值，后端按模型类型分发）

    返回：
      - 成功: {status:'success', fusion{final_score, action, ...}, safety, model_results, ...}
      - 失败: {status:'error', message}
    """
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "请上传图片文件"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "不支持的文件格式"}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        file.save(filepath)
        models_str = request.form.get('models', '')
        models = [m.strip() for m in models_str.split(',') if m.strip()] if models_str else None

        # 解析前端传来的各模型阈值，按模型类型分发
        per_model_thresholds = {}

        # MobileNet 5-class 阈值
        mob_th = {}
        for key in ['porn', 'hentai', 'sexy', 'porn_hentai']:
            val = request.form.get(f'threshold_{key}', type=float)
            if val is not None:
                mob_th[key] = val
        if mob_th:
            per_model_thresholds['mobilenet'] = mob_th

        # 二分类模型阈值（OpenNSFW2 和 Falconsai 共用）
        bin_th = {}
        for key in ['nsfw_block', 'nsfw_review']:
            val = request.form.get(f'threshold_{key}', type=float)
            if val is not None:
                bin_th[key] = val
        if bin_th:
            per_model_thresholds['opennsfw2'] = bin_th
            per_model_thresholds['falconsai'] = bin_th

        result = fusion_detector.detect(
            filepath, models=models,
            thresholds=per_model_thresholds if per_model_thresholds else None,
        )
        return jsonify(result)

    except Exception as e:
        logger.exception("融合检测接口异常")
        return jsonify({"status": "error", "message": f"融合检测失败: {str(e)}"}), 500

    finally:
        # 清理临时文件
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError as e:
            logger.warning("临时文件清理失败 %s: %s", filepath, e)


# ---- 定时检测服务 API ----

@app.route('/api/scheduler/config', methods=['GET'])
def get_scheduler_config():
    """GET /api/scheduler/config — 返回定时服务配置"""
    return jsonify(scheduler_service.get_config_info())


@app.route('/api/scheduler/status', methods=['GET'])
def get_scheduler_status():
    """GET /api/scheduler/status — 返回运行状态 + 执行历史"""
    return jsonify(scheduler_service.get_status())


@app.route('/api/scheduler/trigger', methods=['POST'])
def trigger_scheduler():
    """POST /api/scheduler/trigger — 手动触发一次批次执行"""
    ok, msg = scheduler_service.trigger_manual()
    return jsonify({"success": ok, "message": msg})


# ---- NSFW 检测服务 API（供外部业务调用）----

@app.route('/api/detect/nsfw/check', methods=['POST'])
def detect_nsfw_check():
    """
    POST /api/detect/nsfw/check — 外部业务调用的内容安全检测接口

    参数（JSON body）：
      - imgUrl:         图片网络 URL（必填）
      - modelStrategy:  模型参数策略（可选），格式：
        {
            "modelId": "falconsai",                      # 模型 ID 或 "fusion"
            "models": ["opennsfw2", "falconsai"],        # 融合模式下参与的模型
            "strategy": "weighted_average",               # 融合策略
            "thresholds": { "nsfw_block": 0.8, ... }     # 阈值参数
        }

    返回：
      - 成功: 与 /api/detect/nsfw 或 /api/detect/nsfw/fusion 一致的结果 JSON
      - 失败: {status:'error', message, request_id}
    """
    try:
        # 支持 JSON 和 form 两种提交方式
        if request.is_json:
            data = request.get_json(silent=True) or {}
        else:
            data = request.form.to_dict()

        img_url = data.get('imgUrl', '').strip()
        model_strategy = data.get('modelStrategy')

        # 如果 modelStrategy 是字符串（form 提交），尝试解析为 JSON
        if isinstance(model_strategy, str):
            try:
                model_strategy = json.loads(model_strategy) if model_strategy else None
            except (json.JSONDecodeError, ValueError):
                return jsonify({
                    "status": "error",
                    "message": "modelStrategy 格式无效，请提供合法的 JSON"
                }), 400

        result, status_code = nsfw_service.check(img_url, model_strategy)
        return jsonify(result), status_code

    except Exception as e:
        logger.exception("NSFW 检测服务接口异常")
        return jsonify({
            "status": "error",
            "message": f"接口处理失败: {str(e)}"
        }), 500


# ---- 错误处理 ----

@app.errorhandler(413)
def request_entity_too_large(error):
    """请求体超出大小限制"""
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({"status": "error", "message": f"文件过大，最大支持{max_mb}MB"}), 413


@app.errorhandler(404)
def not_found(error):
    """请求路径不存在"""
    return jsonify({"status": "error", "message": "接口不存在"}), 404


@app.errorhandler(500)
def internal_server_error(error):
    """服务器内部错误"""
    return jsonify({"status": "error", "message": "服务器内部错误"}), 500


if __name__ == '__main__':
    srv = config.get('server', {}) if config else {}
    app.run(
        host=srv.get('host', '0.0.0.0'),
        port=srv.get('port', 5000),
        debug=srv.get('debug', False),
    )
