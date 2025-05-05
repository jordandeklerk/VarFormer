import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    Adds positional encoding to input embeddings.
    
    Args:
        d_model (int): Dimension of the embedding vector.
        max_len (int, optional): Maximum sequence length. Defaults to 5000.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for positional embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Positional encoding up to the length of input.
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Converts input tokens into embeddings using 1D convolution.
    
    Args:
        c_in (int): Number of input channels.
        d_model (int): Dimension of the embedding vector.
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Forward pass for token embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Token embeddings.
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Provides fixed (non-trainable) positional embeddings.
    
    Args:
        c_in (int): Input dimension.
        d_model (int): Dimension of the embedding vector.
    """
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        Forward pass for fixed embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Fixed embeddings (detached from computation graph).
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Combines embeddings for different time features.
    
    Args:
        d_model (int): Dimension of the embedding vector.
        embed_type (str, optional): Type of embedding to use. Defaults to 'fixed'.
        freq (str, optional): Frequency of the time series. Defaults to 'h'.
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        Forward pass for temporal embedding.
        
        Args:
            x (torch.Tensor): Input tensor with time features.
            
        Returns:
            torch.Tensor: Combined temporal embeddings.
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Embeds time features using a linear transformation.
    
    Args:
        d_model (int): Dimension of the embedding vector.
        embed_type (str, optional): Type of embedding. Defaults to 'timeF'.
        freq (str, optional): Frequency of the time series. Defaults to 'h'.
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        Forward pass for time feature embedding.
        
        Args:
            x (torch.Tensor): Input tensor with time features.
            
        Returns:
            torch.Tensor: Embedded time features.
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    Combines value, position, and temporal embeddings.
    
    Args:
        c_in (int): Number of input channels.
        d_model (int): Dimension of the embedding vector.
        embed_type (str, optional): Type of temporal embedding. Defaults to 'fixed'.
        freq (str, optional): Frequency of the time series. Defaults to 'h'.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass for data embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
            x_mark (torch.Tensor, optional): Temporal features. Can be None.
            
        Returns:
            torch.Tensor: Combined embeddings.
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    Provides inverted data embedding using linear transformation.
    
    Args:
        c_in (int): Number of input channels.
        d_model (int): Dimension of the embedding vector.
        embed_type (str, optional): Type of embedding. Defaults to 'fixed'.
        freq (str, optional): Frequency of the time series. Defaults to 'h'.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass for inverted data embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
            x_mark (torch.Tensor, optional): Temporal features. Can be None.
            
        Returns:
            torch.Tensor: Embedded tensor.
        """
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    Combines value and temporal embeddings without positional encoding.
    
    Args:
        c_in (int): Number of input channels.
        d_model (int): Dimension of the embedding vector.
        embed_type (str, optional): Type of temporal embedding. Defaults to 'fixed'.
        freq (str, optional): Frequency of the time series. Defaults to 'h'.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass for data embedding without positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor.
            x_mark (torch.Tensor, optional): Temporal features. Can be None.
            
        Returns:
            torch.Tensor: Combined embeddings without positional encoding.
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Creates and embeds patches from input sequences.
    
    Args:
        d_model (int): Dimension of the embedding vector.
        patch_len (int): Length of each patch.
        stride (int): Stride between patches.
        padding (int): Padding applied to the input sequence.
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for patch embedding.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            tuple: 
                - torch.Tensor: Embedded patches.
                - int: Number of variables in the input.
        """
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars