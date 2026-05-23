import torch
from transformers import AutoFeatureExtractor, AutoTokenizer
from .model_encode import AudioEmbeddingModel, BERTEmbeddingModel, SpeechEmbeddingModel, TextEmbeddingModel

PKL_DIR = "metadata"
OUTPUT_DIR = "features_output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_ENCODER_CONFIGS = {
    "bert-base-uncased": {
        "model_name": "bert-base-uncased",
        "feature_tag": "BERT",
        "legacy": True,
    },
    "phobert-base": {
        "model_name": "vinai/phobert-base",
        "feature_tag": "PHOBERT",
    },
}

AUDIO_ENCODER_CONFIGS = {
    "wav2vec2-base": {
        "model_name": "facebook/wav2vec2-base",
        "feature_tag": "WAV2VEC",
        "legacy": True,
    },
    "hubert-base-ls960": {
        "model_name": "facebook/hubert-base-ls960",
        "feature_tag": "HUBERT",
    },
    "wavlm-base-plus": {
        "model_name": "microsoft/wavlm-base-plus",
        "feature_tag": "WAVLM",
    },
}


def get_feature_prefix(dataset_prefix, text_model_key="bert-base-uncased", audio_model_key="wav2vec2-base"):
    text_cfg = TEXT_ENCODER_CONFIGS[text_model_key]
    audio_cfg = AUDIO_ENCODER_CONFIGS[audio_model_key]
    return "{}_{}_{}".format(dataset_prefix, text_cfg["feature_tag"], audio_cfg["feature_tag"])


def build_text_components(text_model_key="bert-base-uncased"):
    cfg = TEXT_ENCODER_CONFIGS[text_model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)
    if cfg.get("legacy"):
        model = BERTEmbeddingModel(cfg["model_name"]).to(device)
    else:
        model = TextEmbeddingModel(cfg["model_name"]).to(device)
    model.eval()
    return tokenizer, model


def build_audio_components(audio_model_key="wav2vec2-base"):
    cfg = AUDIO_ENCODER_CONFIGS[audio_model_key]
    processor = AutoFeatureExtractor.from_pretrained(cfg["model_name"])
    if cfg.get("legacy"):
        model = AudioEmbeddingModel(cfg["model_name"]).to(device)
    else:
        model = SpeechEmbeddingModel(cfg["model_name"]).to(device)
    model.eval()
    return processor, model

TOKENIZER, TEXT_MODEL = build_text_components()
AUDIO_PROCESSOR, AUDIO_MODEL = build_audio_components()

TEXT_MODEL.eval()
AUDIO_MODEL.eval()
