import math
import copy 
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor


def get_activation_fn(activation: str):
    """
    Retrieves the activation function corresponding to the given name.

    Args:
        activation (str): The name of the activation function. Supported values are
                          "relu", "gelu", and "glu".

    Returns:
        Callable: The corresponding PyTorch activation function.

    Raises:
        RuntimeError: If the activation name is not supported.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """
    Creates a ModuleList containing N deep copies of the given module.

    Args:
        module (nn.Module): The module to be cloned.
        N (int): The number of clones to create.

    Returns:
        nn.ModuleList: A list containing N deep copies of the module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def init_t_xy(end_x: int, end_y: int):
    """
    Initializes t_x and t_y tensors for positional encoding.

    Args:
        end_x (int): The size along the x-axis.
        end_y (int): The size along the y-axis.

    Returns:
        Tuple[Tensor, Tensor]: Two tensors representing the x and y positions respectively.
    """
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> Tensor:
    """
    Computes the axial complex exponential embeddings for rotary positional encoding.

    Args:
        dim (int): The dimensionality of the model.
        end_x (int): The size along the x-axis.
        end_y (int): The size along the y-axis.
        theta (float, optional): Scaling factor for frequency. Defaults to 10000.0.

    Returns:
        Tensor: A tensor containing the concatenated complex embeddings for x and y axes.
    """
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the frequency complex embeddings to be broadcastable with the input tensor.

    Args:
        freqs_cis (torch.Tensor): The frequency complex embeddings.
        x (torch.Tensor): The input tensor to match dimensions with.

    Returns:
        torch.Tensor: The reshaped frequency complex embeddings.
    
    Raises:
        AssertionError: If the input tensor does not have sufficient dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, "Input tensor must have at least 2 dimensions."
    if freqs_cis.shape != (x.shape[-2], x.shape[-1]):
        # Resize freqs_cis by repeating or truncating
        target_shape = (x.shape[-2], x.shape[-1])
        repeat_size = (
            max(1, target_shape[0] // freqs_cis.shape[0]),
            max(1, target_shape[1] // freqs_cis.shape[1])
        )
        freqs_cis = freqs_cis.repeat(repeat_size)
        
        # Truncate if necessary
        freqs_cis = freqs_cis[:target_shape[0], :target_shape[1]]
    
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Applies rotary positional encoding to the query and key tensors.

    Args:
        xq (torch.Tensor): Query tensor of shape (..., seq_len, dim).
        xk (torch.Tensor): Key tensor of shape (..., seq_len, dim). Can be empty.
        freqs_cis (torch.Tensor): Rotary frequency complex embeddings.
        repeat_freqs_k (bool, optional): Whether to repeat frequency embeddings for keys. Defaults to False.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - Rotated query tensor.
            - Rotated key tensor if provided, otherwise the original key tensor.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    
    if xk_ is None:
        return xq_out.type_as(xq).to(xq.device), xk
    
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.

    Args:
        embedding_dim (int): The dimension of the input embeddings.
        num_heads (int): The number of attention heads.
        downsample_rate (int, optional): The factor by which to downsample the embedding dimension. Defaults to 1.
        dropout (float, optional): Dropout probability applied after attention. Defaults to 0.0.
        kv_in_dim (int, optional): The input dimension for keys and values. If None, defaults to embedding_dim. Defaults to None.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """
        Splits the last dimension of the tensor into multiple heads.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, dim).
            num_heads (int): Number of attention heads.

        Returns:
            Tensor: Reshaped tensor of shape (batch, num_heads, seq_len, dim_per_head).
        """
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        Recombines the multiple heads into a single tensor.

        Args:
            x (Tensor): Tensor of shape (batch, num_heads, seq_len, dim_per_head).

        Returns:
            Tensor: Recombined tensor of shape (batch, seq_len, num_heads * dim_per_head).
        """
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Forward pass for the attention layer.

        Args:
            q (Tensor): Query tensor of shape (batch, seq_len, embedding_dim).
            k (Tensor): Key tensor of shape (batch, seq_len, kv_in_dim).
            v (Tensor): Value tensor of shape (batch, seq_len, kv_in_dim).

        Returns:
            Tensor: Output tensor after applying attention and projection, of shape (batch, seq_len, embedding_dim).
        """
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0

        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(nn.Module):
    """
    Rotary Position Embedding (RoPE) based Attention mechanism.

    Args:
        d_model (int): The dimension of the model.
        num_heads (int): The number of attention heads.
        dropout (float, optional): Dropout probability applied to attention weights. Defaults to 0.0.
        rope_theta (float, optional): Scaling factor for RoPE. Defaults to 10000.0.
        rope_k_repeat (bool, optional): Whether to repeat frequency embeddings for keys. Defaults to False.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, rope_theta: float = 10000.0, rope_k_repeat: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rope_theta = rope_theta
        self.rope_k_repeat = rope_k_repeat

    def compute_freqs_cis(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Computes the complex exponential frequencies for rotary positional encoding.

        Args:
            seq_len (int): The sequence length.
            device (torch.device): The device on which to create the tensor.

        Returns:
            Tensor: The complex frequencies tensor of shape (seq_len, head_dim).
        """
        theta = self.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)).to(device)
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def apply_rotary_pos_emb(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        """
        Applies rotary positional embeddings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch, num_heads, seq_len, head_dim).
            freqs_cis (Tensor): Rotary frequency complex embeddings.

        Returns:
            Tensor: The tensor after applying rotary positional embeddings.
        """
        x_rot = torch.complex(x[..., ::2], x[..., 1::2])
        x_out = torch.view_as_real(x_rot * freqs_cis).flatten(-2)
        return x_out

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for the RoPE-based attention.

        Args:
            q (Tensor): Query tensor of shape (batch, num_heads, seq_len, head_dim).
            k (Tensor): Key tensor of shape (batch, num_heads, seq_len, head_dim).
            v (Tensor): Value tensor of shape (batch, num_heads, seq_len, head_dim).
            attn_mask (Optional[Tensor], optional): Attention mask tensor. Defaults to None.

        Returns:
            Tensor: Output tensor after applying RoPE attention, of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = q.shape
        device = q.device

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute freqs_cis
        max_seq_len = max(q.size(2), k.size(2), v.size(2))
        freqs_cis = self.compute_freqs_cis(max_seq_len, device)
        freqs_cis = freqs_cis.view(1, 1, max_seq_len, -1)

        # Apply rotary position encoding
        q_rot = self.apply_rotary_pos_emb(q, freqs_cis[:, :, :q.size(2), :])
        k_rot = self.apply_rotary_pos_emb(k, freqs_cis[:, :, :k.size(2), :])

        # Attention
        attn_weights = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_weights += attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(out)


class MemoryAttentionLayer(nn.Module):
    """
    A memory attention layer that combines self-attention and cross-attention mechanisms.

    Args:
        activation (str): Activation function to use ("relu" or "gelu").
        cross_attention (nn.Module): Cross-attention module.
        d_model (int): The dimension of the model.
        dim_feedforward (int): The dimension of the feedforward network.
        dropout (float): Dropout probability.
        pos_enc_at_attn (bool): Whether to apply positional encoding at attention.
        pos_enc_at_cross_attn_keys (bool): Whether to apply positional encoding to cross-attention keys.
        pos_enc_at_cross_attn_queries (bool): Whether to apply positional encoding to cross-attention queries.
        self_attention (nn.Module): Self-attention module.
    """

    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt: Tensor, query_pos: Optional[Tensor], tgt_mask: Optional[Tensor]) -> Tensor:
        """
        Forward pass for the self-attention sublayer.

        Args:
            tgt (Tensor): Target tensor of shape (batch, seq_len, d_model).
            query_pos (Optional[Tensor]): Positional encoding for queries.
            tgt_mask (Optional[Tensor]): Attention mask for the target.

        Returns:
            Tensor: Updated target tensor after self-attention.
        """
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2, attn_mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt: Tensor, memory: Tensor, query_pos: Optional[Tensor], pos: Optional[Tensor]) -> Tensor:
        """
        Forward pass for the cross-attention sublayer.

        Args:
            tgt (Tensor): Target tensor of shape (batch, seq_len, d_model).
            memory (Tensor): Memory tensor from the encoder of shape (batch, mem_seq_len, d_model).
            query_pos (Optional[Tensor]): Positional encoding for queries.
            pos (Optional[Tensor]): Positional encoding for keys.

        Returns:
            Tensor: Updated target tensor after cross-attention.
        """
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for the MemoryAttentionLayer.

        Args:
            tgt (Tensor): Target tensor of shape (batch, seq_len, d_model).
            memory (Tensor): Memory tensor from the encoder of shape (batch, mem_seq_len, d_model).
            pos (Optional[Tensor], optional): Positional encoding for memory. Defaults to None.
            query_pos (Optional[Tensor], optional): Positional encoding for target queries. Defaults to None.
            tgt_mask (Optional[Tensor], optional): Attention mask for the target. Defaults to None.

        Returns:
            Tensor: Updated target tensor after applying self-attention, cross-attention, and feedforward network.
        """
        tgt = self._forward_sa(tgt, query_pos, tgt_mask)
        tgt = self._forward_ca(tgt, memory, query_pos, pos)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt