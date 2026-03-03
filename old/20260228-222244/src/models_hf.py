#/home/lyclyq/Optimization/grad-shake-align/src/models_hf.py

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


def build_model(model_name: str, num_labels: int) -> torch.nn.Module:
    cfg = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
