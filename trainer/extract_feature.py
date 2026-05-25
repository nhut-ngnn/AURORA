import os
import sys
import torch
import pickle
import soundfile as sf
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_extract.config import (
    PKL_DIR, OUTPUT_DIR, device,
    AUDIO_ENCODER_CONFIGS,
    TEXT_ENCODER_CONFIGS,
    build_audio_components,
    build_text_components,
    get_feature_prefix,
)
from src.architecture.ASR_model import SpeechToText, _load_waveform
from src.architecture.configs.asr_models import ASR_MODEL_CONFIGS, DEFAULT_ASR_MODEL_KEY

DATASET_PREFIXES = {
    "IEMOCAP": "IEMOCAP",
    "MSP-IMPROV": "MSPIMPROV",
}

DATASET_METADATA_DIRS = {
    "IEMOCAP": "IEMOCAP_preprocessed",
    "MSP-IMPROV": "MSP-IMPROV_preprocessed",
}

MIN_AUDIO_SAMPLES = 400  

def extract_text_features(text, tokenizer, model, device):
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
        pooled = output[0] if isinstance(output, tuple) else output
        return pooled.squeeze().cpu()
    except Exception as e:
        print(f"[TEXT] Error: {e}")
        return None

_speech_to_text = None
_asr_model_key = DEFAULT_ASR_MODEL_KEY
TOKENIZER = None
TEXT_MODEL = None
AUDIO_PROCESSOR = None
AUDIO_MODEL = None

def _set_asr_model_key(model_key):
    global _asr_model_key, _speech_to_text
    if model_key != _asr_model_key:
        _asr_model_key = model_key
        _speech_to_text = None

def _get_speech_to_text():
    global _speech_to_text
    if _speech_to_text is None:
        _speech_to_text = SpeechToText(device=device, model_key=_asr_model_key)
    return _speech_to_text

def _clean_dataset_text(text):
    if isinstance(text, str):
        cleaned = text.strip()
        return cleaned if cleaned else None
    return None

def _get_asr_text(audio_path):
    stt = _get_speech_to_text()
    return stt.transcribe(audio_path)

def _set_embedding_models(text_model_key, audio_model_key):
    global TOKENIZER, TEXT_MODEL, AUDIO_PROCESSOR, AUDIO_MODEL
    TOKENIZER, TEXT_MODEL = build_text_components(text_model_key)
    AUDIO_PROCESSOR, AUDIO_MODEL = build_audio_components(audio_model_key)

def _prepare_words_for_wer(text):
    if not isinstance(text, str):
        return None
    cleaned = text.strip().lower()
    return cleaned.split() if cleaned else None

def _levenshtein_distance(ref_words, hyp_words):
    m, n = len(ref_words), len(hyp_words)
    if m == 0:
        return None
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      
                curr[j - 1] + 1,  
                prev[j - 1] + cost
            )
        prev = curr
    return prev[-1]

def compute_wer_components(reference, hypothesis):
    ref_words = _prepare_words_for_wer(reference)
    hyp_words = _prepare_words_for_wer(hypothesis)
    if not ref_words:
        return None, None
    if hyp_words is None:
        hyp_words = []
    distance = _levenshtein_distance(ref_words, hyp_words)
    if distance is None:
        return None, None
    return distance, len(ref_words)

def extract_audio_features(audio_path, processor, model, device):
    try:
        waveform, sr = _load_waveform(audio_path) 
        if waveform.numel() < MIN_AUDIO_SAMPLES:
            print(f"[AUDIO] Too short, skipping {audio_path} (samples={waveform.numel()})")
            return None
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
        with torch.no_grad():
            output = model(input_values, attention_mask=attention_mask)
        pooled = output[0] if isinstance(output, tuple) else output
        return pooled.squeeze().cpu()
    except Exception as e:
        print(f"[AUDIO] Error in {audio_path}: {e}")
        return None

def process_single_sample(audio_path, text, label):
    dataset_text = _clean_dataset_text(text)
    asr_text = _get_asr_text(audio_path)

    dataset_text_embed = extract_text_features(dataset_text, TOKENIZER, TEXT_MODEL, device)
    asr_text_embed = extract_text_features(asr_text, TOKENIZER, TEXT_MODEL, device)

    primary_text_embed = dataset_text_embed if dataset_text_embed is not None else asr_text_embed
    audio_embed = extract_audio_features(audio_path, AUDIO_PROCESSOR, AUDIO_MODEL, device)

    if primary_text_embed is not None and audio_embed is not None:
        return {
            'text_embed': primary_text_embed,
            'text_embed_dataset': dataset_text_embed,
            'text_embed_asr': asr_text_embed,
            'text_raw_dataset': dataset_text,
            'text_raw_asr': asr_text,
            'audio_embed': audio_embed,
            'audio_path': audio_path,
            'label': torch.tensor(label) if isinstance(label, (int, float)) else label,
            'sample_id': f"{os.path.basename(audio_path)}"
        }
    else:
        return None

def process_dataset(pkl_path, output_path, split_name=None):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    processed_samples = []
    print(f"Processing {len(data)} samples from {pkl_path}")
    wer_edits, wer_ref_words, wer_count = 0, 0, 0

    for idx, (audio_path, text, label) in tqdm(enumerate(data), total=len(data)):
        sample = process_single_sample(audio_path, text, label)
        if sample is not None:
            processed_samples.append(sample)
            if split_name == "test":
                dist, ref_len = compute_wer_components(sample["text_raw_dataset"], sample["text_raw_asr"])
                if dist is not None and ref_len:
                    wer_edits += dist
                    wer_ref_words += ref_len
                    wer_count += 1
        else:
            print(f"[SKIP] Failed to process: {audio_path}")

    with open(output_path, "wb") as f:
        pickle.dump(processed_samples, f)

    print(f"Saved processed data to: {output_path}")
    print(f"Total processed samples: {len(processed_samples)}")
    if split_name == "test":
        if wer_ref_words > 0:
            wer_value = wer_edits / wer_ref_words
            print(f"[TEST WER] {wer_value:.4f} over {wer_count} samples ({wer_ref_words} reference words)")
        else:
            print("[TEST WER] Not enough reference transcripts to compute WER.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_PREFIXES.keys()),
        help="Dataset to process",
    )
    parser.add_argument(
        "--asr_model",
        type=str,
        default=DEFAULT_ASR_MODEL_KEY,
        choices=sorted(ASR_MODEL_CONFIGS.keys()),
        help="ASR model key for transcription.",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="bert-base-uncased",
        choices=sorted(TEXT_ENCODER_CONFIGS.keys()),
        help="Text encoder for dataset and ASR transcripts.",
    )
    parser.add_argument(
        "--audio_model",
        type=str,
        default="wav2vec2-base",
        choices=sorted(AUDIO_ENCODER_CONFIGS.keys()),
        help="Audio encoder for acoustic features.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output folder (default: features_output/<DATASET>_ASR_<model_key>).",
    )
    args = parser.parse_args()

    _set_asr_model_key(args.asr_model)
    _set_embedding_models(args.text_model, args.audio_model)
    print("Starting feature extraction...")
    print(f"Using device: {device}")
    print(f"ASR model: {args.asr_model}")
    print(f"Text encoder: {args.text_model}")
    print(f"Audio encoder: {args.audio_model}")

    pkl_prefix = DATASET_PREFIXES[args.dataset]
    metadata_dir = DATASET_METADATA_DIRS[args.dataset]

    asr_tag = args.asr_model.replace("/", "_")
    output_dir = args.output_dir or os.path.join(OUTPUT_DIR, f"{args.dataset}_ASR_{asr_tag}")
    os.makedirs(output_dir, exist_ok=True)
    feature_prefix = get_feature_prefix(pkl_prefix, args.text_model, args.audio_model)

    datasets = [
        ("train", f"{metadata_dir}/train.pkl", f"{feature_prefix}_train.pkl"),
        ("val",   f"{metadata_dir}/val.pkl",   f"{feature_prefix}_val.pkl"),
        ("test",  f"{metadata_dir}/test.pkl",  f"{feature_prefix}_test.pkl")
    ]

    for split_name, pkl_file, output_file in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split: {pkl_file}")
        print(f"{'='*50}")
        process_dataset(
            os.path.join(PKL_DIR, pkl_file),
            os.path.join(output_dir, output_file),
            split_name=split_name
        )

if __name__ == "__main__":
    main()
