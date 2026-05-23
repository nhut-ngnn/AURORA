DEFAULT_ASR_MODEL_KEY = "whisper-large-v2"

ASR_MODEL_CONFIGS = {
    "wav2vec2-base-100h": {
        "model_name": "facebook/wav2vec2-base-100h",
        "model_type": "ctc",
    },
    "wav2vec2-base-960h": {
        "model_name": "facebook/wav2vec2-base-960h",
        "model_type": "ctc",
    },
    "wav2vec2-large-960h": {
        "model_name": "facebook/wav2vec2-large-960h",
        "model_type": "ctc",
    },
    "wav2vec2-large-960h-lv60-self": {
        "model_name": "facebook/wav2vec2-large-960h-lv60-self",
        "model_type": "ctc",
    },
    "hubert-large-ls960-ft": {
        "model_name": "facebook/hubert-large-ls960-ft",
        "model_type": "ctc",
        "low_cpu_mem_usage": False,

    },
    "wavlm-base-plus": {
        "model_name": "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
        "model_type": "ctc",
        "low_cpu_mem_usage": False,
    },
    "wavlm-libri-clean-100h-base-plus": {
        "model_name": "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
        "model_type": "ctc",
        "low_cpu_mem_usage": False,
    },
    "whisper-tiny.en": {
        "model_name": "openai/whisper-tiny.en",
        "model_type": "seq2seq",
    },
    "whisper-base.en": {
        "model_name": "openai/whisper-base.en",
        "model_type": "seq2seq",
    },
    "whisper-small.en": {
        "model_name": "openai/whisper-small.en",
        "model_type": "seq2seq",
    },
    "whisper-medium.en": {
        "model_name": "openai/whisper-medium.en",
        "model_type": "seq2seq",
    },
    "whisper-large-v2": {
        "model_name": "openai/whisper-large-v2",
        "model_type": "seq2seq",
    },
    "whisper-large-v2.en": {
        "model_name": "openai/whisper-large-v2",
        "model_type": "seq2seq",
    },
    "whisper-large-v3": {
        "model_name": "openai/whisper-large-v3",
        "model_type": "openai_whisper",
        "openai_model_name": "large-v3",
    },
    "nvidia/parakeet-tdt-0.6b-v2": {
        "model_name": "nvidia/parakeet-tdt-0.6b-v2",
        "model_type": "seq2seq",
    },
    "nvidia/canary-180m-flash": {
        "model_name": "nvidia/canary-180m-flash",
        "model_type": "seq2seq",
    },
    "facebook/seamless-m4t-v2-large": {
        "model_name": "facebook/seamless-m4t-v2-large",
        "model_type": "seq2seq",
    },
}
