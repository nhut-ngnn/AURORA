from torch import nn
from transformers import AutoModel, BertModel, Wav2Vec2Model

class BERTEmbeddingModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.bert(**kwargs)
        hidden_states = outputs.last_hidden_state  
        pooled_output = hidden_states.mean(dim=1)
        return pooled_output


class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        hidden_states = outputs.last_hidden_state
        if attention_mask is None:
            return hidden_states.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom


class AudioEmbeddingModel(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)

    def forward(self, input_values, attention_mask=None):
        kwargs = {"input_values": input_values}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self.wav2vec(**kwargs)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        return pooled_output


class SpeechEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_values, attention_mask=None):
        kwargs = {"input_values": input_values}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self.encoder(**kwargs)
        hidden_states = outputs.last_hidden_state
        return hidden_states.mean(dim=1)
