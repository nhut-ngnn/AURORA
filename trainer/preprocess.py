import argparse
import glob
import logging
import os
import pickle
import random
import re
import soundfile as sf
import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABEL_MAP_4 = {
    "ang": 0,
    "hap": 1,
    "sad": 2,
    "neu": 3,
    "exc": 1 
}

LABEL_MAP_MSP_IMPROV = {
    "A": 0,  # Angry
    "H": 1,  # Happy
    "S": 2,  # Sad
    "N": 3,  # Neutral
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _safe_stratified_split(samples, labels, test_size, seed):
    try:
        return train_test_split(samples, labels, test_size=test_size, random_state=seed, stratify=labels)
    except ValueError:
        logging.warning("Falling back to non-stratified split because at least one class is too small.")
        return train_test_split(samples, labels, test_size=test_size, random_state=seed)


def _split_train_val_test_prompt4mer(samples, labels, seed, test_size=0.1, val_size=0.1):
    train_val, test_samples, train_val_labels, _ = _safe_stratified_split(
        samples, labels, test_size=test_size, seed=seed
    )
    val_ratio = val_size / (1.0 - test_size)
    train_samples, val_samples, _, _ = _safe_stratified_split(
        train_val, train_val_labels, test_size=val_ratio, seed=seed
    )
    return train_samples, val_samples, test_samples


def preprocess_iemocap(args):
    session_ids = list(range(1, 6))
    ignore_length = args.ignore_length
    seed = args.seed
    data_root = args.data_root

    label_map = LABEL_MAP_4
    valid_emotions = {"ang", "hap", "sad", "neu", "exc"}

    samples = []
    labels = []

    for sess_id in tqdm.tqdm(session_ids, desc="Processing IEMOCAP"):
        sess_path = os.path.join(data_root, f"Session{sess_id}")
        audio_root = os.path.join(sess_path, "sentences/wav")
        text_root = os.path.join(sess_path, "dialog/transcriptions")
        label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        label_files = glob.glob(os.path.join(label_root, "*.txt"))

        for label_file in label_files:
            base_name = os.path.basename(label_file)
            transcript_file = os.path.join(text_root, base_name)

            with open(transcript_file, "r") as f:
                transcript_lines = {
                    line.split(":")[0]: line.split(":")[1].strip()
                    for line in f.readlines()
                }

            with open(label_file, "r") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    data = line[1:].split()
                    start_time = float(data[0])
                    end_time = float(data[2][:-1])
                    utt_id = data[3]
                    emotion = data[4]

                    if emotion not in valid_emotions:
                        continue

                    folder = utt_id[:-5]
                    wav_name = utt_id + ".wav"
                    wav_path = os.path.join(audio_root, folder, wav_name)

                    try:
                        wav_data, _ = sf.read(wav_path, dtype="int16")
                    except Exception:
                        logging.warning(f"Cannot read {wav_path}")
                        continue

                    if len(wav_data) < ignore_length:
                        logging.warning(f"Ignored short sample: {wav_path}")
                        continue

                    text_key = f"{utt_id} [{start_time:08.4f}-{end_time:08.4f}]"
                    text = transcript_lines.get(text_key)

                    if text is None:
                        text_key_alt1 = f"{utt_id} [{start_time:08.4f}-{end_time + 0.0001:08.4f}]"
                        text_key_alt2 = f"{utt_id} [{start_time + 0.0001:08.4f}-{end_time:08.4f}]"
                        text = transcript_lines.get(text_key_alt1) or transcript_lines.get(text_key_alt2)

                    if text is None:
                        logging.warning(f"Transcript not found: {text_key}")
                        continue

                    label = label_map.get(emotion)
                    if label is None:
                        continue

                    samples.append((wav_path, text, label))
                    labels.append(label)

    data = list(zip(samples, labels))
    random.Random(seed).shuffle(data)

    if not data:
        logging.error(
            "No IEMOCAP samples were collected. Check data_root, ignore_length, and that transcripts match labels."
        )
        return

    samples, labels = zip(*data)

    train, test_samples, train_labels, _ = train_test_split(
        samples, labels, test_size=0.1, random_state=seed
    )
    train_samples, val_samples, _, _ = train_test_split(
        train, train_labels, test_size=0.1, random_state=seed
    )

    output_dir = "metadata/IEMOCAP_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logging.info(f"Saved preprocessed data to {output_dir}")

def preprocess_msp_improv(args):
    data_root = args.data_root
    ignore_length = args.ignore_length
    seed = args.seed

    audio_root = os.path.join(data_root, "Audio")
    transcript_root = os.path.join(data_root, "Human_transcriptions", "All_human_transcriptions")
    wav_files = sorted(glob.glob(os.path.join(audio_root, "**", "*.wav"), recursive=True))
    samples = []
    labels = []

    if not wav_files:
        logging.error(f"No MSP-IMPROV wav files found under {audio_root}")
        return

    for wav_path in tqdm.tqdm(wav_files, desc="Processing MSP-IMPROV"):
        sample_id = os.path.splitext(os.path.basename(wav_path))[0]
        match = re.match(r"MSP-IMPROV-S\d{2}([AHSN])-", sample_id)
        if match is None:
            logging.warning(f"Cannot parse emotion label from filename: {sample_id}")
            continue

        label = LABEL_MAP_MSP_IMPROV.get(match.group(1))
        if label is None:
            continue

        transcript_path = os.path.join(transcript_root, f"{sample_id}.txt")
        if not os.path.exists(transcript_path):
            logging.warning(f"Transcript not found: {transcript_path}")
            continue

        try:
            wav_data, _ = sf.read(wav_path, dtype="int16")
        except Exception:
            logging.warning(f"Cannot read {wav_path}")
            continue

        if len(wav_data) < ignore_length:
            logging.warning(f"Ignored short sample: {wav_path}")
            continue

        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            text = " ".join(line.strip() for line in f if line.strip())
        if not text:
            logging.warning(f"Empty transcript: {transcript_path}")
            continue

        samples.append((wav_path, text, label))
        labels.append(label)

    if not samples:
        logging.error("No MSP-IMPROV samples were collected. Check data_root and transcript files.")
        return

    data = list(zip(samples, labels))
    random.Random(seed).shuffle(data)
    samples, labels = zip(*data)

    train_samples, val_samples, test_samples = _split_train_val_test_prompt4mer(
        list(samples), list(labels), seed=seed
    )

    output_dir = "metadata/MSP-IMPROV_preprocessed"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    logging.info(f"Saved preprocessed data to {output_dir}")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["IEMOCAP", "MSP-IMPROV"], required=True)
    parser.add_argument("--data_root", type=str, required=True, help="Root path to dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ignore_length", type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    if args.dataset == "IEMOCAP":
        preprocess_iemocap(args)
    elif args.dataset == "MSP-IMPROV":
        preprocess_msp_improv(args)
