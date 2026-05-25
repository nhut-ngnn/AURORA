# AURORA

AURORA is a multimodal speech emotion recognition repository. The model combines acoustic embeddings and text embeddings, supports ASR-derived transcripts, and trains with a teacher-student objective so the student can learn from clean transcript features while remaining usable with ASR text at test time.

## Highlights

- Multimodal emotion recognition from audio and text features.
- ASR-aware feature extraction with multiple ASR backends.
- Teacher-student training with clean-text teacher and ASR-text student.
- Audio-guided gated fusion, cross-modal encoders, uncertainty gating, and residual text repair.
- Evaluation utilities for audio + clean text and audio + ASR text test modes.
- Dataset preprocessing support for IEMOCAP and MSP-IMPROV.

## Repository Structure

```text
AURORA/
├── metadata/              # Preprocessed train/val/test metadata
├── features_output/       # Extracted audio/text/ASR feature pickle files
├── saved_model/           # Trained checkpoints
├── src/
│   ├── architecture/      # AURORA model and component modules
│   ├── feature_extract/   # Embedding model configuration and encoders
│   ├── training_process/  # Training loop and IO helpers
│   └── utils/             # Metrics and evaluation helpers
└── trainer/               # CLI scripts for preprocessing, features, training, evaluation
```

## Installation

Create a Python environment, then install the main dependencies:

```bash
pip install torch torchaudio transformers soundfile tqdm numpy pandas scikit-learn wandb
```

Some ASR backends may require extra packages or model downloads depending on the selected ASR model.

## Data Preparation

Preprocess raw datasets into `metadata/<DATASET>_preprocessed/{train,val,test}.pkl`.

```bash
python trainer/preprocess.py --dataset IEMOCAP --data_root /path/to/IEMOCAP
python trainer/preprocess.py --dataset MSP-IMPROV --data_root /path/to/MSP-IMPROV
```

Each metadata sample stores an audio path, transcript text when available, and an emotion label.

## Feature Extraction

Extract audio embeddings, clean transcript embeddings, ASR transcript embeddings, and raw ASR text:

```bash
python trainer/extract_feature.py --dataset IEMOCAP --asr_model whisper-large-v2
```

By default, features are written to:

```text
features_output/<DATASET>_ASR_<ASR_MODEL>/
```

Supported text encoders include `bert-base-uncased` and `phobert-base`. Supported audio encoders include `wav2vec2-base`, `hubert-base-ls960`, and `wavlm-base-plus`.

Example with explicit encoders:

```bash
python trainer/extract_feature.py \
  --dataset IEMOCAP \
  --asr_model whisper-large-v2 \
  --text_model bert-base-uncased \
  --audio_model wav2vec2-base
```

## Training

Train AURORA across the default seeds:

```bash
python trainer/train.py --dataset IEMOCAP --asr_model whisper-large-v2
```

Useful options:

```bash
python trainer/train.py \
  --dataset IEMOCAP \
  --asr_model whisper-large-v2 \
  --epochs 30 \
  --stage1_epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --lambda_kd 1.0 \
  --temperature 2.0 \
  --selection_metric val_loss
```

Checkpoints are saved under:

```text
saved_model/<DATASET>_ASR_<ASR_MODEL>/AURORA_<DATASET>_seed<SEED>.pt
saved_model/<DATASET>_ASR_<ASR_MODEL>/AURORA_stage1_<DATASET>_seed<SEED>.pt
```

To disable online Weights & Biases logging:

```bash
python trainer/train.py --dataset IEMOCAP --asr_model whisper-large-v2 --wandb_mode disabled
```

## Evaluation

Evaluate checkpoints on both clean-text and ASR-text test modes:

```bash
python trainer/evaluate_test_modes.py --dataset IEMOCAP --asr_model whisper-large-v2
```

Save summaries and per-sample predictions:

```bash
python trainer/evaluate_test_modes.py \
  --dataset IEMOCAP \
  --asr_model whisper-large-v2 \
  --save_summary_csv results/summary.csv \
  --save_predictions_csv results/predictions.csv
```

Inspect correctly predicted samples where ASR text differs from clean text:

```bash
python trainer/predict.py \
  --dataset IEMOCAP \
  --asr_model whisper-large-v2 \
  --max_samples 20 \
  --sort_by_wer
```

## Ablation

Run ablation experiments with:

```bash
python trainer/ablation.py --dataset IEMOCAP --asr_model whisper-large-v2
```

Use `--help` on any script to view the available arguments.

## Model Import

The main model class lives in:

```python
from src.architecture.AURORA import AURORA
```

## License

This project is released under the license included in `LICENSE`.
