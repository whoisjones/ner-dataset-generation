import json
import argparse
from pathlib import Path
from typing import Dict, Any

from transformers import PretrainedConfig

class SpanModelConfig(PretrainedConfig):

    def __init__(
        self,
        token_encoder: str = None,
        type_encoder: str = None,
        type_encoder_pooling: str = "cls",
        loss_fn: str = "bce",
        max_span_length: int = 30,
        linear_hidden_size: int = 128,
        span_width_embedding_size: int = 128,
        dropout: float = 0.1,
        init_temperature: float = 0.03,
        prediction_threshold: float = 0.5,
        bce_start_pos_weight: float = None,
        bce_end_pos_weight: float = None,
        bce_span_pos_weight: float = None,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        contrastive_threshold_loss_weight: float = 0.5,
        contrastive_span_loss_weight: float = 0.5,
        contrastive_tau: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_encoder = token_encoder
        self.type_encoder = type_encoder
        self.loss_fn = loss_fn
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.max_span_length = max_span_length
        self.dropout = dropout
        self.linear_hidden_size = linear_hidden_size
        self.span_width_embedding_size = span_width_embedding_size
        self.init_temperature = init_temperature
        self.type_encoder_pooling = type_encoder_pooling
        self.prediction_threshold = prediction_threshold
        self.bce_start_pos_weight = bce_start_pos_weight
        self.bce_end_pos_weight = bce_end_pos_weight
        self.bce_span_pos_weight = bce_span_pos_weight
        self.contrastive_threshold_loss_weight = contrastive_threshold_loss_weight
        self.contrastive_span_loss_weight = contrastive_span_loss_weight
        self.contrastive_tau = contrastive_tau