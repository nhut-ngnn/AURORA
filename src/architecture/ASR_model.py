import torch
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC

from src.architecture.configs.asr_models import ASR_MODEL_CONFIGS, DEFAULT_ASR_MODEL_KEY


def _load_waveform(path: str):
    audio_np, sr = sf.read(path, dtype="float32")
    waveform = torch.tensor(audio_np, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.transpose(0, 1)
    else:
        raise ValueError(f"Unexpected audio shape {waveform.shape} for {path}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        gcd = np.gcd(sr, 16000)
        up = 16000 // gcd
        down = sr // gcd
        resampled = resample_poly(waveform.squeeze(0).numpy(), up, down).astype(np.float32)
        waveform = torch.from_numpy(resampled).unsqueeze(0)
        sr = 16000

    return waveform, sr


class SpeechToText:
    def __init__(
        self,
        device=None,
        language="en",
        task="transcribe",
        model_key=None,
        model_name=None,
        model_type=None,
        model_revision=None,
        processor_revision=None,
        model_subfolder=None,
        processor_subfolder=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.task = task
        self.model_key = model_key or DEFAULT_ASR_MODEL_KEY

        config = ASR_MODEL_CONFIGS.get(self.model_key)
        if config is None and model_name is not None:
            config = {"model_name": model_name, "model_type": model_type or "seq2seq"}
        elif config is None:
            config = ASR_MODEL_CONFIGS[DEFAULT_ASR_MODEL_KEY]

        self.model_name = config["model_name"]
        self.model_type = config["model_type"]
        self.processor_name = config.get("processor_name", self.model_name)
        self.openai_model_name = config.get("openai_model_name")
        self.model_revision = model_revision or config.get("model_revision") or config.get("revision")
        self.processor_revision = processor_revision or config.get("processor_revision") or config.get("revision")
        self.model_revisions = config.get("model_revisions") or config.get("revisions")
        self.processor_revisions = config.get("processor_revisions") or config.get("revisions")
        self.model_subfolder = model_subfolder or config.get("model_subfolder") or config.get("subfolder")
        self.processor_subfolder = processor_subfolder or config.get("processor_subfolder") or config.get("subfolder")
        self.model_subfolders = config.get("model_subfolders") or config.get("subfolders")
        self.processor_subfolders = config.get("processor_subfolders") or config.get("subfolders")
        self.low_cpu_mem_usage = config.get("low_cpu_mem_usage", True)
        self.torch_dtype = (
            torch.float16 if (self.device == "cuda" and self.model_type == "seq2seq") else torch.float32
        )

        if self.model_type == "openai_whisper":
            try:
                import whisper
            except ImportError as exc:
                raise ImportError("openai-whisper is required for Whisper v3. Install with `pip install -U openai-whisper`.") from exc
            self._whisper = whisper
            model_name = self.openai_model_name or self.model_name
            self.model = whisper.load_model(model_name, device=self.device)
            self.model.eval()
            self.processor = None
            return

        def _revision_candidates(primary, fallback_list):
            candidates = []
            if primary:
                candidates.append(primary)
            if fallback_list:
                for value in fallback_list:
                    if value not in candidates:
                        candidates.append(value)
            candidates.append(None)
            return candidates

        def _subfolder_candidates(primary, fallback_list):
            candidates = []
            if primary:
                candidates.append(primary)
            if fallback_list:
                for value in fallback_list:
                    if value not in candidates:
                        candidates.append(value)
            candidates.append(None)
            return candidates

        def _load_processor(name, revisions, subfolders):
            attempts = [
                {},
                {"use_fast": False},
                {"use_fast": False, "legacy": False},
                {"legacy": False},
            ]
            last_exc = None
            for subfolder in subfolders:
                for revision in revisions:
                    for kwargs in attempts:
                        try:
                            load_kwargs = dict(kwargs)
                            if revision is not None:
                                load_kwargs["revision"] = revision
                            if subfolder is not None:
                                load_kwargs["subfolder"] = subfolder
                            return AutoProcessor.from_pretrained(name, **load_kwargs)
                        except TypeError:
                            continue
                        except (ValueError, OSError) as exc:
                            last_exc = exc
                            if "Wrong index found" in str(exc) or "added_tokens" in str(exc):
                                continue
                            if "config.json" in str(exc):
                                continue
                            if "not a valid git identifier" in str(exc):
                                continue
                            break
            if last_exc is not None:
                raise last_exc
            return AutoProcessor.from_pretrained(name)

        try:
            self.processor = _load_processor(
                self.processor_name,
                _revision_candidates(self.processor_revision, self.processor_revisions),
                _subfolder_candidates(self.processor_subfolder, self.processor_subfolders),
            )
        except ValueError as exc:
            if self.processor_name != self.model_name and "Wrong index found" in str(exc):
                self.processor = _load_processor(
                    self.model_name,
                    _revision_candidates(self.processor_revision, self.processor_revisions),
                    _subfolder_candidates(self.processor_subfolder, self.processor_subfolders),
                )
            else:
                raise

        def _load_model(model_cls, revisions, subfolders):
            kwargs = {
                "torch_dtype": self.torch_dtype,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }
            for subfolder in subfolders:
                for revision in revisions:
                    for use_safetensors in (True, False):
                        try:
                            load_kwargs = dict(kwargs)
                            if revision is not None:
                                load_kwargs["revision"] = revision
                            if subfolder is not None:
                                load_kwargs["subfolder"] = subfolder
                            return model_cls.from_pretrained(
                                self.model_name,
                                use_safetensors=use_safetensors,
                                **load_kwargs,
                            )
                        except TypeError:
                            load_kwargs = dict(kwargs)
                            if revision is not None:
                                load_kwargs["revision"] = revision
                            if subfolder is not None:
                                load_kwargs["subfolder"] = subfolder
                            return model_cls.from_pretrained(self.model_name, **load_kwargs)
                        except OSError as exc:
                            if use_safetensors and "safetensors" in str(exc):
                                continue
                            if "config.json" in str(exc):
                                continue
                            if "not a valid git identifier" in str(exc):
                                continue
                            raise
            return model_cls.from_pretrained(self.model_name, **kwargs)

        if self.model_type == "ctc":
            self.model = _load_model(
                AutoModelForCTC,
                _revision_candidates(self.model_revision, self.model_revisions),
                _subfolder_candidates(self.model_subfolder, self.model_subfolders),
            ).to(self.device)
        else:
            self.model = _load_model(
                AutoModelForSpeechSeq2Seq,
                _revision_candidates(self.model_revision, self.model_revisions),
                _subfolder_candidates(self.model_subfolder, self.model_subfolders),
            ).to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: str):
        try:
            waveform, sr = _load_waveform(audio_path)
            audio = waveform.squeeze(0).cpu().numpy().astype(np.float32)

            if self.model_type == "openai_whisper":
                whisper = self._whisper
                audio = whisper.pad_or_trim(audio)
                n_mels = getattr(self.model.dims, "n_mels", 128)
                mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(self.model.device)
                options = whisper.DecodingOptions(
                    language=self.language,
                    task=self.task,
                    fp16=(self.device == "cuda"),
                )
                result = whisper.decode(self.model, mel, options)
                return result.text.strip()

            if self.model_type == "ctc":
                inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
                input_values = inputs.input_values.to(self.device, dtype=self.torch_dtype)
                attention_mask = None
                if "attention_mask" in inputs:
                    attention_mask = inputs.attention_mask.to(self.device)
                with torch.no_grad():
                    logits = self.model(input_values, attention_mask=attention_mask).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                return self.processor.batch_decode(predicted_ids)[0].strip()

            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

            gen_kwargs = {}
            if self.language is not None:
                gen_kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                    language=self.language,
                    task=self.task,
                )

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, **gen_kwargs)

            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        except Exception as exc:
            print(f"[Whisper S2T] Failed to transcribe {audio_path}: {exc}")
            return None
