import logging
import torch
import torch.nn as nn

logger = logging.getLogger('aws')


class LSTMModel(nn.Module):
    """Stacked LSTM with dropout for time series regression.

    Inspired by Rogendo/forex-lstm-models architecture:
    2 LSTM layers with dropout -> Linear head -> scalar output.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.num_features = config.get('num_features', 20)
        self.hidden_size = config.get('lstm_hidden_size', 64)
        self.num_layers = config.get('lstm_num_layers', 2)
        self.dropout = config.get('lstm_dropout', 0.2)

        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # x: [batch, lookback, features]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_size]
        out = self.dropout_layer(last_hidden)
        out = self.fc(out)  # [batch, 1]
        return out


class TransformerModel(nn.Module):
    """Transformer encoder for time series regression.

    Inspired by SatyamSinghal/financial-ttm architecture:
    Input projection -> Positional encoding -> TransformerEncoder ->
    Global average pooling -> Linear head -> scalar output.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.num_features = config.get('num_features', 20)
        self.d_model = config.get('transformer_d_model', 64)
        self.nhead = config.get('transformer_nhead', 4)
        self.num_layers = config.get('transformer_num_layers', 2)
        self.dim_feedforward = config.get('transformer_dim_ff', 128)
        self.dropout = config.get('transformer_dropout', 0.1)
        lookback = config.get('lookback', 20)

        self.input_projection = nn.Linear(self.num_features, self.d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, lookback, self.d_model) * 0.1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, x):
        # x: [batch, lookback, features]
        x = self.input_projection(x)       # [batch, lookback, d_model]
        x = x + self.pos_encoding           # add positional encoding
        x = self.transformer_encoder(x)     # [batch, lookback, d_model]
        x = x.mean(dim=1)                   # global average pool -> [batch, d_model]
        out = self.fc(x)                     # [batch, 1]
        return out


class BiLSTMAttentionModel(nn.Module):
    """Bidirectional LSTM with self-attention for time series regression.

    Inspired by JonusNattapong/xauusd-trading-ai architecture:
    BiLSTM -> Self-attention (scoring + weighted sum) -> Linear head -> scalar output.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.num_features = config.get('num_features', 20)
        self.hidden_size = config.get('bilstm_hidden_size', 64)
        self.num_layers = config.get('bilstm_num_layers', 2)
        self.dropout = config.get('bilstm_dropout', 0.2)

        self.bilstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        # Attention: project BiLSTM output (2*hidden) to attention scores
        self.attention_fc = nn.Linear(self.hidden_size * 2, 1)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):
        # x: [batch, lookback, features]
        lstm_out, _ = self.bilstm(x)  # [batch, lookback, hidden*2]

        # Compute attention weights
        attn_scores = self.attention_fc(lstm_out)        # [batch, lookback, 1]
        attn_weights = torch.softmax(attn_scores, dim=1) # [batch, lookback, 1]

        # Weighted sum of BiLSTM outputs
        context = (lstm_out * attn_weights).sum(dim=1)   # [batch, hidden*2]

        out = self.dropout_layer(context)
        out = self.fc(out)  # [batch, 1]
        return out


MODEL_REGISTRY = {
    'lstm': LSTMModel,
    'transformer': TransformerModel,
    'bilstm_attention': BiLSTMAttentionModel,
}


def get_model(model_name: str, config: dict) -> nn.Module:
    """Factory function to create a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](config)
