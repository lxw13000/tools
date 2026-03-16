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
import uuid
import logging
import yaml
from .modules import MotionDetector, NSFWDetector, FusionDetector

logger = logging.getLogger(__name__)

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
        logger.warning("无法加载配置文件 %s, 使用默认配置. 错误: %s", config_path, e)
        return None


config = load_config()

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
    """GET /api/nsfw/config — 返回 MobileNet 默认阈值（供前端渲染滑块初始值）"""
    return jsonify(nsfw_detector.get_default_thresholds())


@app.route('/api/motion/config', methods=['GET'])
def get_motion_config():
    """GET /api/motion/config — 返回动态检测默认权重和阈值（供前端渲染滑块初始值）"""
    return jsonify({
        "weights": motion_detector.weights,
        "thresholds": motion_detector.thresholds,
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
        for key in ['static', 'dynamic']:
            val = request.form.get(f'threshold_{key}', type=float)
            if val is not None:
                thresholds[key] = val

        # 调用动态检测器
        return jsonify(motion_detector.detect(
            saved_paths,
            weights=weights if weights else None,
            thresholds=thresholds if thresholds else None,
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


# ---- 错误处理 ----

@app.errorhandler(413)
def request_entity_too_large(error):
    """请求体超出大小限制"""
    max_mb = app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    return jsonify({"status": "error", "message": f"文件过大，最大支持{max_mb}MB"}), 413


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
