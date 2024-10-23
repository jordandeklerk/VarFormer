from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

from Embed import *
from Memory_Attention import *


class PositionwiseFeedForward(nn.Module):
    """
    Implements a position-wise feedforward network.

    Args:
        d_model (int): Dimension of the input and output.
        d_ff (int): Dimension of the hidden layer.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the feedforward network to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor: Output tensor after applying the feedforward network.
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Defines a single layer of the Transformer encoder.

    Args:
        d_model (int): Dimension of the input and output.
        d_ff (int): Dimension of the feedforward network.
        n_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = Attention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).
            src_mask (Optional[Tensor], optional): Source attention mask. Defaults to None.

        Returns:
            Tensor: Output tensor after applying self-attention and feedforward network.
        """
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))


class DecoderLayer(nn.Module):
    """
    Defines a single layer of the Transformer decoder with memory attention.

    Args:
        d_model (int): Dimension of the input and output.
        d_ff (int): Dimension of the feedforward network.
        n_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """

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
        """
        Forward pass for the decoder layer.

        Args:
            x (Tensor): Input tensor of shape (batch, tgt_seq_len, d_model).
            memory (Tensor): Memory tensor from the encoder of shape (batch, src_seq_len, d_model).
            src_pos (Optional[Tensor], optional): Positional encoding for source memory. Defaults to None.
            tgt_pos (Optional[Tensor], optional): Positional encoding for target queries. Defaults to None.
            tgt_mask (Optional[Tensor], optional): Attention mask for the target. Defaults to None.

        Returns:
            Tensor: Output tensor after applying memory attention.
        """
        x = self.norm1(self.self_memory_attention(x, x, pos=tgt_pos, query_pos=tgt_pos, tgt_mask=tgt_mask))
        x = self.norm2(self.cross_memory_attention(x, memory, pos=src_pos, query_pos=tgt_pos))
        return x


class Encoder(nn.Module):
    """
    Transformer encoder composed of multiple encoder layers.

    Args:
        d_model (int): Dimension of the input and output.
        d_ff (int): Dimension of the feedforward network.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the encoder.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor: Output tensor after passing through all encoder layers and normalization.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Transformer decoder composed of multiple decoder layers.

    Args:
        d_model (int): Dimension of the input and output.
        d_ff (int): Dimension of the feedforward network.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of decoder layers.
        dropout (float): Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ff, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor, src_pos: Optional[Tensor] = None, tgt_pos: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the decoder.

        Args:
            x (Tensor): Input tensor of shape (batch, tgt_seq_len, d_model).
            memory (Tensor): Memory tensor from the encoder of shape (batch, src_seq_len, d_model).
            src_pos (Optional[Tensor], optional): Positional encoding for source memory. Defaults to None.
            tgt_pos (Optional[Tensor], optional): Positional encoding for target queries. Defaults to None.
            tgt_mask (Optional[Tensor], optional): Attention mask for the target. Defaults to None.

        Returns:
            Tensor: Output tensor after passing through all decoder layers and normalization.
        """
        for layer in self.layers:
            x = layer(x, memory, src_pos, tgt_pos, tgt_mask)
        return self.norm(x)


class TransformerModel(nn.Module):
    """
    Transformer-based model for sequence-to-sequence tasks with forecasting capabilities.

    Args:
        params: A parameter object containing model configurations such as:
                - c_in: Number of input channels.
                - d_model: Dimension of the embeddings.
                - embed_type: Type of embedding ('fixed', 'learnable', or 'timeF').
                - freq: Frequency type ('h', 't', etc.).
                - dropout: Dropout probability.
                - d_ff: Dimension of the feedforward network.
                - n_heads: Number of attention heads.
                - e_layers: Number of encoder layers.
                - d_layers: Number of decoder layers.
                - c_out: Number of output channels.
                - pred_len: Prediction length.
                - seq_len: Sequence length.
    """

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
        """
        Creates a causal mask for autoregressive decoding to prevent attention to future positions.

        Args:
            sz (int): Size of the mask (sequence length).

        Returns:
            Tensor: Causal mask tensor of shape (1, sz, sz).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.unsqueeze(0)  # Add batch dimension

    def forward(self, src: Tensor, src_mark: Tensor, tgt: Tensor, tgt_mark: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Transformer model.

        Args:
            src (Tensor): Source input tensor of shape (batch, src_seq_len, c_in).
            src_mark (Tensor): Source temporal features of shape (batch, src_seq_len, features).
            tgt (Tensor): Target input tensor of shape (batch, tgt_seq_len, c_in).
            tgt_mark (Tensor): Target temporal features of shape (batch, tgt_seq_len, features).

        Returns:
            Tuple[Tensor, Tensor]: 
                - mu: Predicted mean tensor of shape (batch, tgt_seq_len, c_out).
                - sigma: Predicted variance tensor of shape (batch, tgt_seq_len, c_out).
        """
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
        """
        Generates forecasts by iteratively predicting future time steps.

        Args:
            src (Tensor): Source input tensor of shape (batch, src_seq_len, c_in).
            src_mark (Tensor): Source temporal features of shape (batch, src_seq_len, features).
            tgt (Tensor): Target input tensor of shape (batch, tgt_seq_len, c_in).
            tgt_mark (Tensor): Target temporal features of shape (batch, tgt_seq_len, features).

        Returns:
            Tuple[Tensor, Tensor]: 
                - mu: Forecasted mean tensor of shape (batch, pred_len, c_out).
                - sigma: Forecasted variance tensor of shape (batch, pred_len, c_out).
        """
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
