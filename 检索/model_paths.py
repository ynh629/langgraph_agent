# model_paths.py
"""模型路径解析：优先使用 ModelScope 预下载的本地副本，绕开 HuggingFace 网络依赖。

配合 scripts/download_models.py 使用：模型从 ModelScope 下载到
~/.cache/huggingface/local/<model_id>/（容器内即 hf_cache 共享卷），
运行时通过 resolve_model() 解析为本地目录离线加载。
"""
import os

# ModelScope 预下载模型的本地存放根目录（容器内挂载在 hf_cache 卷）
LOCAL_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "local")


def resolve_model(model_id: str) -> str:
    """把 HuggingFace 模型 ID 解析为实际加载路径。

    解析顺序：已是本地目录 → 原样返回；本地缓存存在 → 缓存路径；
    否则返回原始 ID（退回 sentence-transformers 默认行为，走 HF_ENDPOINT）。
    """
    if os.path.isdir(model_id):
        return model_id
    local = os.path.join(LOCAL_ROOT, model_id)
    if os.path.isdir(local):
        return local
    return model_id
