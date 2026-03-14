#!/usr/bin/env python3
"""
下载 NSFW 分类模型

3 个模型:
  1. OpenNSFW2 (Yahoo)     — pip install opennsfw2，首次推理自动下载 (~23MB)
  2. MobileNet V2 140      — GantMan 5 分类，从 GitHub Release 下载 (~17MB)
  3. Falconsai ViT          — HuggingFace snapshot_download (~330MB)

使用方法: python download_models.py
"""

import os
import sys
import subprocess
import shutil
import zipfile
import tempfile

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


def download(url, target):
    """用 curl 下载文件"""
    print(f'  -> {url}')
    cmd = ['curl', '-L', '-f', '--progress-bar', '-o', target, url]
    try:
        result = subprocess.run(cmd, timeout=600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print('  超时')
        return False
    except Exception as e:
        print(f'  异常: {e}')
        return False


def download_from_zip(zip_url, zip_entry, target):
    """从远程 zip 中提取指定文件"""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not download(zip_url, tmp_path):
            return False

        with zipfile.ZipFile(tmp_path) as z:
            with z.open(zip_entry) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        return True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def validate(path, min_mb):
    """检查文件是否有效"""
    if not os.path.exists(path):
        return False
    size_mb = os.path.getsize(path) / 1024 / 1024
    return size_mb >= min_mb


def validate_falconsai(model_dir):
    """检查 Falconsai 模型目录是否有效"""
    config_path = os.path.join(model_dir, 'config.json')
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    bin_path = os.path.join(model_dir, 'pytorch_model.bin')
    return os.path.exists(config_path) and (
        os.path.exists(safetensors_path) or os.path.exists(bin_path)
    )


def download_mobilenet_v2_140():
    """下载 MobileNet V2 140 (GantMan 5-class)"""
    filename = 'mobilenet_v2_140_224.h5'
    target = os.path.join(MODELS_DIR, filename)

    if validate(target, 10):
        size_mb = os.path.getsize(target) / 1024 / 1024
        print(f'[已存在] MobileNet V2 140  ({size_mb:.1f}MB)')
        return True

    if os.path.exists(target):
        os.remove(target)

    print('[下载中] MobileNet V2 140 (v1.2, 224x224)')
    ok = download_from_zip(
        'https://github.com/GantMan/nsfw_model/releases/download/1.2.0/mobilenet_v2_140_224.1.zip',
        'mobilenet_v2_140_224/saved_model.h5',
        target,
    )

    if ok and validate(target, 10):
        size_mb = os.path.getsize(target) / 1024 / 1024
        print(f'  [完成] {size_mb:.1f}MB\n')
        return True
    else:
        if os.path.exists(target):
            os.remove(target)
        print(f'  [失败] 请手动下载到: {target}')
        print(f'  参考: https://github.com/GantMan/nsfw_model\n')
        return False


def download_falconsai():
    """下载 Falconsai/nsfw_image_detection 模型"""
    falconsai_dir = os.path.join(MODELS_DIR, 'falconsai')

    if validate_falconsai(falconsai_dir):
        print('[已存在] Falconsai ViT')
        return True

    print('[下载中] Falconsai ViT (nsfw_image_detection)')
    os.makedirs(falconsai_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id='Falconsai/nsfw_image_detection',
            local_dir=falconsai_dir,
            allow_patterns=[
                'config.json',
                'preprocessor_config.json',
                'model.safetensors',
                'pytorch_model.bin',
            ],
        )
        if validate_falconsai(falconsai_dir):
            print('  [完成] Falconsai ViT\n')
            return True
    except ImportError:
        print('  需要 huggingface-hub: pip install huggingface-hub')
    except Exception as e:
        print(f'  异常: {e}')

    print(f'  [失败] 请手动下载到: {falconsai_dir}')
    print(f'  参考: https://huggingface.co/Falconsai/nsfw_image_detection\n')
    return False


def prewarm_opennsfw2():
    """预热 OpenNSFW2 模型（首次 import 会自动下载 ~23MB 权重）"""
    print('[预热中] OpenNSFW2 (Yahoo)')
    try:
        import opennsfw2 as n2
        n2.make_open_nsfw_model()
        print('  [完成] OpenNSFW2 模型已缓存\n')
        return True
    except ImportError:
        print('  需要 opennsfw2: pip install opennsfw2')
    except Exception as e:
        print(f'  异常: {e}')

    print('  [失败] OpenNSFW2 将在首次请求时自动下载\n')
    return False


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f'模型目录: {MODELS_DIR}\n')

    results = []

    # 1. MobileNet V2 140 (需要 curl)
    if shutil.which('curl'):
        results.append(download_mobilenet_v2_140())
    else:
        print('[跳过] MobileNet V2 140 — 需要 curl')
        results.append(False)

    # 2. Falconsai ViT
    results.append(download_falconsai())

    # 3. OpenNSFW2 预热
    results.append(prewarm_opennsfw2())

    success = sum(results)
    total = len(results)
    print(f'结果: {success}/{total} 个模型可用')
    return 0 if success > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
