"""
Flask API 服务主文件
提供图片分析的 REST API 接口
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import yaml
from .modules import MotionDetector, NSFWDetector, FusionDetector

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}, 使用默认配置. 错误: {str(e)}")
        return None


config = load_config()

if config:
    server_config = config.get('server', {})
    upload_config = config.get('upload', {})
    app.config['MAX_CONTENT_LENGTH'] = server_config.get('max_content_length', 52428800)
    app.config['UPLOAD_FOLDER'] = upload_config.get('folder', '/tmp/uploads')
    app.config['ALLOWED_EXTENSIONS'] = set(upload_config.get('allowed_extensions',
                                           ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp']))
else:
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化检测器 — 只需 2 个对象
motion_config = config.get('motion_detection', {}) if config else {}
motion_detector = MotionDetector(
    similarity_threshold=motion_config.get('similarity_threshold', 0.95)
)
nsfw_detector = NSFWDetector(
    models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'),
    config=config
)
fusion_detector = FusionDetector(nsfw_detector=nsfw_detector, config=config)


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ---- 页面路由 ----

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/motion')
def motion_page():
    return render_template('motion.html')


@app.route('/nsfw')
def nsfw_page():
    return render_template('nsfw.html')


# ---- API ----

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "服务运行正常"})


@app.route('/api/nsfw/models', methods=['GET'])
def get_nsfw_models():
    return jsonify(nsfw_detector.get_models_info())


@app.route('/api/nsfw/config', methods=['GET'])
def get_nsfw_config():
    return jsonify(nsfw_detector.get_default_thresholds())


@app.route('/api/detect/motion', methods=['POST'])
def detect_motion():
    if 'images' not in request.files:
        return jsonify({"status": "error", "message": "请上传图片文件"}), 400

    files = request.files.getlist('images')
    if not files or len(files) == 0:
        return jsonify({"status": "error", "message": "请至少上传一张图片"}), 400

    saved_paths = []
    session_id = str(uuid.uuid4())

    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{session_id}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                saved_paths.append(filepath)

        if not saved_paths:
            return jsonify({"status": "error", "message": "没有有效的图片文件"}), 400

        return jsonify(motion_detector.detect(saved_paths))

    except Exception as e:
        return jsonify({"status": "error", "message": f"处理失败: {str(e)}"}), 500

    finally:
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass


@app.route('/api/detect/nsfw', methods=['POST'])
def detect_nsfw():
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

        # 根据模型类型解析阈值
        thresholds = {}
        if model_id == 'mobilenet':
            for key in ['porn', 'hentai', 'sexy', 'porn_hentai']:
                val = request.form.get(f'threshold_{key}', type=float)
                if val is not None:
                    thresholds[key] = val
        else:
            for key in ['nsfw_block', 'nsfw_review']:
                val = request.form.get(f'threshold_{key}', type=float)
                if val is not None:
                    thresholds[key] = val

        result = nsfw_detector.detect(
            filepath,
            model_id=model_id,
            thresholds=thresholds if thresholds else None
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": f"处理失败: {str(e)}"}), 500

    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass


@app.route('/api/detect/nsfw/fusion', methods=['POST'])
def detect_nsfw_fusion():
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
        result = fusion_detector.detect(filepath, models=models)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": f"融合检测失败: {str(e)}"}), 500

    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"status": "error", "message": "文件过大，最大支持50MB"}), 413


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"status": "error", "message": "服务器内部错误"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
