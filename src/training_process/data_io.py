import os
import pickle

from src.architecture.configs.asr_models import DEFAULT_ASR_MODEL_KEY

DEFAULT_FEATURE_DIR = "features_output"
DEFAULT_MODEL_DIR = "saved_model"


def normalize_asr_tag(model_key):
    key = model_key or DEFAULT_ASR_MODEL_KEY
    return key.replace("/", "_")


def resolve_feature_dir(base_dir, dataset_tag, asr_model_key=None):
    root = base_dir or DEFAULT_FEATURE_DIR
    if not asr_model_key:
        return root
    suffix = f"{dataset_tag}_ASR_{normalize_asr_tag(asr_model_key)}"
    if os.path.basename(root) == suffix:
        return root
    return os.path.join(root, suffix)

def build_feature_paths(base_dir, dataset_tag, feature_prefix, asr_model_key=None):
    root = resolve_feature_dir(base_dir, dataset_tag, asr_model_key)
    return {
        "train": os.path.join(root, f"{feature_prefix}_train.pkl"),
        "val": os.path.join(root, f"{feature_prefix}_val.pkl"),
        "test": os.path.join(root, f"{feature_prefix}_test.pkl"),
    }


def resolve_checkpoint_dir(base_dir, dataset_tag, asr_model_key=None):
    root = base_dir or DEFAULT_MODEL_DIR
    if not asr_model_key:
        return root
    suffix = f"{dataset_tag}_ASR_{normalize_asr_tag(asr_model_key)}"
    if os.path.basename(root) == suffix:
        return root
    return os.path.join(root, suffix)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
