import os
import torch
from torch.utils.data import Dataset, DataLoader

from src.training_process.data_io import build_feature_paths, load_pkl

DATASET_CONFIG = {
    "IEMOCAP": {"num_classes": 4, "feature_prefix": "IEMOCAP_BERT_WAV2VEC"},
    "ESD": {"num_classes": 5, "feature_prefix": "ESD_BERT_WAV2VEC"},
    "MSP-IMPROV": {"num_classes": 4, "feature_prefix": "MSPIMPROV_BERT_WAV2VEC"},
}


class FeatureDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for item in data:
            clean = item.get("text_embed_dataset")
            if clean is None:
                clean = item.get("text_embed")
            asr = item.get("text_embed_asr")
            if asr is None:
                asr = clean
            if clean is None:
                continue
            self.samples.append(
                {
                    "audio": item["audio_embed"],
                    "clean_text": clean,
                    "asr_text": asr,
                    "label": item["label"],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio = torch.as_tensor(s["audio"])
        clean = torch.as_tensor(s["clean_text"])
        asr = torch.as_tensor(s["asr_text"])
        label = torch.tensor(s["label"], dtype=torch.long)
        return audio, clean, asr, label


def get_dataloaders(args):
    cfg = DATASET_CONFIG[args.dataset]
    feature_prefix = getattr(args, "feature_prefix", None) or cfg["feature_prefix"]
    paths = build_feature_paths(
        args.data_dir,
        args.dataset,
        feature_prefix,
        getattr(args, "asr_model", None),
    )

    loaders = {}
    for split, path in paths.items():
        data = load_pkl(path)
        ds = FeatureDataset(data)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"))
    return loaders
