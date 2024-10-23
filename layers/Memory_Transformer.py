from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

from Embed import *
from Memory_Attention import *


class PositionwiseFeedForward(nn.Module):
    """Position-wise feedforward network with linear transformations and ReLU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Applies feedforward network to input tensor."""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """Single layer of the Transformer encoder with self-attention and feedforward network."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = Attention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """Processes input through self-attention and feedforward network."""
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    """Single layer of the Transformer decoder with self and cross memory attention."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_memory_attention = MemoryAttentionLayer(
            activation="relu",
            cross_attention=RoPEAttention(d_model, n_heads, dropout=dropout),
            d_model=d_model,
            dim_feedforward=d_ff,
            dropout=dropout,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=True,
            self_attention=RoPEAttention(d_model, n_heads, dropout=dropout)
        )
        self.cross_memory_attention = MemoryAttentionLayer(
            activation="relu",
            cross_attention=RoPEAttention(d_model, n_heads, dropout=dropout, rope_k_repeat=True),
            d_model=d_model,
            dim_feedforward=d_ff,
            dropout=dropout,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=True,
            self_attention=RoPEAttention(d_model, n_heads, dropout=dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor, src_pos: Optional[Tensor] = None, tgt_pos: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None) -> Tensor:
        """Processes input through self and cross memory attention."""
        x = self.norm1(self.self_memory_attention(x, x, pos=tgt_pos, query_pos=tgt_pos, tgt_mask=tgt_mask))
        x = self.norm2(self.cross_memory_attention(x, memory, pos=src_pos, query_pos=tgt_pos))
        return x


class Encoder(nn.Module):
    """Stack of Transformer encoder layers."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Processes input through all encoder layers."""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of Transformer decoder layers."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor, src_pos: Optional[Tensor] = None, tgt_pos: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None) -> Tensor:
        """Processes input through all decoder layers."""
        for layer in self.layers:
            x = layer(x, memory, src_pos, tgt_pos, tgt_mask)
        return self.norm(x)


class TransformerModel(nn.Module):
    """Transformer-based sequence-to-sequence model with forecasting capabilities."""

    def __init__(self, params):
        super().__init__()
        self.data_embedding = DataEmbedding(params.c_in, params.d_model, params.embed_type, params.freq, params.dropout)
        self.encoder = Encoder(params.d_model, params.d_ff, params.n_heads, params.e_layers, params.dropout)
        self.decoder = Decoder(params.d_model, params.d_ff, params.n_heads, params.d_layers, params.dropout)
        self.generator_mu = nn.Linear(params.d_model, params.c_out)
        self.generator_sigma = nn.Linear(params.d_model, params.c_out)
        self.pred_len = params.pred_len
        self.d_model = params.d_model
        self.n_heads = params.n_heads

        self.memory = nn.Parameter(torch.randn(1, params.seq_len, params.d_model))
        self.positional_embedding = PositionalEmbedding(params.d_model)

    def create_causal_mask(self, sz: int) -> Tensor:
        """Creates causal mask for autoregressive decoding."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.unsqueeze(0)  # Add batch dimension

    def forward(self, src: Tensor, src_mark: Tensor, tgt: Tensor, tgt_mark: Tensor) -> Tuple[Tensor, Tensor]:
        """Processes input through encoder and decoder to generate predictions."""
        B, L = src.shape[0], src.shape[1]
        S = tgt.shape[1]
        device = src.device

        src_pos = self.positional_embedding(src)
        tgt_pos = self.positional_embedding(tgt)
        
        tgt_mask = self.create_causal_mask(S).to(device)

        src_embedded = self.data_embedding(src, src_mark)
        tgt_embedded = self.data_embedding(tgt, tgt_mark)
        
        enc_output = self.encoder(src_embedded)
        dec_output = self.decoder(tgt_embedded, enc_output, src_pos, tgt_pos, tgt_mask)
        
        mu = self.generator_mu(dec_output)
        sigma = F.softplus(self.generator_sigma(dec_output))  
        return mu, sigma

    def forecast(self, src: Tensor, src_mark: Tensor, tgt: Tensor, tgt_mark: Tensor) -> Tuple[Tensor, Tensor]:
        """Generates forecasts by iteratively predicting future time steps."""
        B, L = src.shape[0], src.shape[1]
        S = tgt.shape[1]
        device = src.device
        
        src_pos = self.positional_embedding(src)
        
        src_embedded = self.data_embedding(src, src_mark)
        enc_output = self.encoder(src_embedded)

        outputs = []
        for i in range(self.pred_len):
            tgt_pos = self.positional_embedding(tgt)
            tgt_mask = self.create_causal_mask(S).to(device)
            
            tgt_embedded = self.data_embedding(tgt, tgt_mark)
            dec_output = self.decoder(tgt_embedded, enc_output, src_pos, tgt_pos, tgt_mask)
            
            mu = self.generator_mu(dec_output[:, -1:, :])
            sigma = F.softplus(self.generator_sigma(dec_output[:, -1:, :]))
            outputs.append((mu, sigma))

            tgt = torch.cat([tgt, mu], dim=1)
            tgt_mark = torch.cat([tgt_mark, tgt_mark[:, -1:, :]], dim=1)
            S += 1

        mu = torch.cat([out[0] for out in outputs], dim=1)
        sigma = torch.cat([out[1] for out in outputs], dim=1)

        return mu, sigma
