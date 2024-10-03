import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) module for normalizing and denormalizing input features.

    This module normalizes input data based on the computed mean and standard deviation,
    and optionally applies learnable affine transformations. It supports both normalization
    ('norm') and denormalization ('denorm') modes, allowing for reversible transformations.

    Args:
        num_features (int): Number of features in the input.
        eps (float, optional): Small value to add to variance for numerical stability. Default is 1e-5.
        affine (bool, optional): If True, applies learnable affine transformation after normalization. Default is True.
        subtract_last (bool, optional): If True, subtracts the last time step instead of the mean. Default is False.

    Attributes:
        num_features (int): Number of input features.
        eps (float): Epsilon value for numerical stability.
        affine (bool): Flag to apply affine transformation.
        subtract_last (bool): Flag to subtract the last time step.
        mean (torch.Tensor or None): Computed mean of the input.
        stdev (torch.Tensor or None): Computed standard deviation of the input.
        last (torch.Tensor or None): Last time step of the input (used if subtract_last is True).
        affine_weight (nn.Parameter): Learnable weight for affine transformation.
        affine_bias (nn.Parameter): Learnable bias for affine transformation.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.mean = None
        self.stdev = None
        self.last = None
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        Forward pass for RevIN.

        Depending on the mode, the input is either normalized or denormalized.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            mode (str): Operation mode, either 'norm' for normalization or 'denorm' for denormalization.

        Returns:
            torch.Tensor: The normalized or denormalized tensor.

        Raises:
            RuntimeError: If denormalization is attempted before normalization.
            NotImplementedError: If an unsupported mode is provided.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            if self.mean is None or self.stdev is None:
                raise RuntimeError("Statistics not computed. Call forward with mode='norm' first.")
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")
        return x

    def _init_params(self):
        """
        Initializes the learnable affine parameters.

        Creates `affine_weight` and `affine_bias` as trainable parameters.
        """
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Computes the mean and standard deviation of the input tensor.

        If `subtract_last` is True, stores the last time step for subtraction instead of the mean.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        Normalizes the input tensor using the computed statistics.

        Applies optional affine transformation after normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Normalized (and possibly affine-transformed) tensor.
        """
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        Denormalizes the input tensor using the stored statistics.

        Reverses the normalization and affine transformation applied during normalization.

        Args:
            x (torch.Tensor): Normalized tensor of shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    

class ResidualBlock(nn.Module):
    """
    Residual Block module consisting of two linear layers with a LeakyReLU activation.

    This block adds the input to the output of a sequence of linear transformations,
    enabling residual connections that help in training deep networks.

    Args:
        in_features (int): Number of input and output features.

    Attributes:
        block (nn.Sequential): Sequential container of linear layers and activations.
    """
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x):
        """
        Forward pass for the Residual Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after adding the residual connection.
        """
        return x + self.block(x)


class Encoder(nn.Module):
    """
    Encoder module for the autoencoder architecture.

    Processes historical sequences and prediction parameters to generate a latent representation.

    Args:
        params (Namespace): Configuration parameters containing model dimensions and sequence lengths.
        revin (RevIN): Instance of the RevIN module for input normalization.

    Attributes:
        revin (RevIN): Reversible Instance Normalization module.
        params (Namespace): Model configuration parameters.
        his_dim (int): Dimension of the flattened historical input.
        his_proj (nn.Linear): Linear layer projecting historical input to intermediate dimension.
        pred_proj (nn.Linear): Linear layer projecting prediction parameters to intermediate dimension.
        encoder_common (nn.Sequential): Sequential container of linear layers and residual blocks for encoding.
    """
    def __init__(self, params, revin):
        super().__init__()
        self.revin = revin
        self.params = params

        self.his_dim = params.c_in * params.seq_len
        self.his_proj = nn.Linear(self.his_dim, params.inter_dim // 2)
        self.pred_proj = nn.Linear(params.c_in * 2, params.inter_dim // 2)

        self.encoder_common = nn.Sequential(
            nn.Linear(params.inter_dim, params.inter_dim),
            nn.LeakyReLU(),
            ResidualBlock(params.inter_dim),
            nn.LeakyReLU(),
            ResidualBlock(params.inter_dim),
            nn.LeakyReLU(),
            ResidualBlock(params.inter_dim),
            nn.LeakyReLU(),
            nn.Linear(params.inter_dim, params.latent_dim * 2)
        )

    def forward(self, his_seq, pred_mu, pred_sigma):
        """
        Forward pass for the Encoder.

        Combines historical sequence and prediction parameters to produce latent mean and log variance.

        Args:
            his_seq (torch.Tensor): Historical sequence tensor of shape (batch_size, seq_len, c_in).
            pred_mu (torch.Tensor): Prediction mean tensor of shape (batch_size, pred_len, c_in).
            pred_sigma (torch.Tensor): Prediction standard deviation tensor of shape (batch_size, pred_len, c_in).

        Returns:
            torch.Tensor: Encoded tensor containing concatenated mean and log variance of the latent space.
        """
        batch_size = his_seq.size(0)
        pred_len = pred_mu.size(1)

        his_seq_flat = his_seq.view(batch_size, -1)
        his_encoded = self.his_proj(his_seq_flat)

        pred_combined = torch.cat([pred_mu, pred_sigma], dim=-1)
        pred_combined_flat = pred_combined.view(batch_size, pred_len, -1)
        pred_encoded = self.pred_proj(pred_combined_flat)
        pred_encoded = pred_encoded.mean(dim=1)

        combined = torch.cat([his_encoded, pred_encoded], dim=1)
        output = self.encoder_common(combined)
        return output


class Decoder(nn.Module):
    """
    Decoder module for the autoencoder architecture.

    Reconstructs the sequence from the latent representation, separating it into trend and seasonal components.

    Args:
        params (Namespace): Configuration parameters containing model dimensions and sequence lengths.
        revin (RevIN): Instance of the RevIN module for input normalization.

    Attributes:
        revin (RevIN): Reversible Instance Normalization module.
        params (Namespace): Model configuration parameters.
        his_dim (int): Dimension of the flattened historical input.
        his_proj (nn.Linear): Linear layer projecting historical input to latent dimension.
        decoder_common (nn.Sequential): Sequential container of linear layers and residual blocks for decoding.
        shared_layer (nn.Sequential): Shared residual layers for trend and season projections.
        season_proj (nn.Sequential): Sequential container for projecting to seasonal components.
        trend_proj (nn.Sequential): Sequential container for projecting to trend components.
    """
    def __init__(self, params, revin):
        super().__init__()
        self.revin = revin
        self.params = params

        self.his_dim = params.c_in * params.seq_len
        self.his_proj = nn.Linear(self.his_dim, params.latent_dim)
        input_dim = params.latent_dim * 2
        hidden_dim = params.inter_dim * 2

        self.decoder_common = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim),
            nn.LeakyReLU()
        )

        self.shared_layer = nn.Sequential(
            ResidualBlock(hidden_dim),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim),
            nn.LeakyReLU()
        )

        self.season_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, params.pred_len * params.c_in)
        )

        self.trend_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, params.pred_len * params.c_in)
        )

    def forward(self, hidden_z, his_seq):
        """
        Forward pass for the Decoder.

        Reconstructs the sequence from the latent representation and historical sequence.

        Args:
            hidden_z (torch.Tensor): Latent representation tensor of shape (batch_size, latent_dim).
            his_seq (torch.Tensor): Historical sequence tensor of shape (batch_size, seq_len, c_in).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reconstructed sequence of shape (batch_size, pred_len, c_in).
                - Reconstructed trend component of shape (batch_size, pred_len, c_in).
                - Reconstructed seasonal component of shape (batch_size, pred_len, c_in).
        """
        batch_size = hidden_z.size(0)
        his_seq_flat = his_seq.view(batch_size, -1)
        his_seq_proj = self.his_proj(his_seq_flat)

        combined = torch.cat([hidden_z, his_seq_proj], dim=1)
        common = self.decoder_common(combined)

        shared_features = self.shared_layer(common)

        season = self.season_proj(shared_features)
        trend = self.trend_proj(shared_features)

        season = season.view(batch_size, self.params.pred_len, self.params.c_in)
        trend = trend.view(batch_size, self.params.pred_len, self.params.c_in)

        output = season + trend
        return output, trend, season


class AutoEncoder(nn.Module):
    """
    Variational Autoencoder (VAE) for sequence data.

    Combines the Encoder and Decoder with a Reversible Instance Normalization module.
    Implements the reparameterization trick for sampling from the latent distribution.

    Args:
        params (Namespace): Configuration parameters containing model dimensions and sequence lengths.

    Attributes:
        revin (RevIN): Reversible Instance Normalization module.
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        params (Namespace): Model configuration parameters.
    """
    def __init__(self, params):
        super().__init__()
        self.revin = RevIN(num_features=params.c_in)
        self.encoder = Encoder(params, self.revin)
        self.decoder = Decoder(params, self.revin)
        self.params = params

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from the latent distribution.

        Args:
            mu (torch.Tensor): Mean of the latent distribution of shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log variance of the latent distribution of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Sampled latent vector of shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, his_seq, pred_mu, pred_sigma):
        """
        Forward pass for the AutoEncoder.

        Normalizes the input, encodes it into the latent space, samples from the latent distribution,
        and decodes the latent vector back to the sequence space.

        Args:
            his_seq (torch.Tensor): Historical sequence tensor of shape (batch_size, seq_len, c_in).
            pred_mu (torch.Tensor): Prediction mean tensor of shape (batch_size, pred_len, c_in).
            pred_sigma (torch.Tensor): Prediction standard deviation tensor of shape (batch_size, pred_len, c_in).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reconstructed sequence of shape (batch_size, seq_len, c_in).
                - Reconstructed trend component of shape (batch_size, pred_len, c_in).
                - Reconstructed seasonal component of shape (batch_size, pred_len, c_in).
                - Mean of the latent distribution of shape (batch_size, latent_dim).
                - Log variance of the latent distribution of shape (batch_size, latent_dim).
        """
        his_seq_norm = self.revin(his_seq, mode='norm')
        enc = self.encoder(his_seq_norm, pred_mu, pred_sigma)
        mu, logvar = enc.chunk(2, dim=-1)
        
        hidden_z = self.reparameterize(mu, logvar)
        recon_seq, recon_trend, recon_season = self.decoder(hidden_z, his_seq_norm)
        
        recon_seq = self.revin(recon_seq, mode='denorm')
        return recon_seq, recon_trend, recon_season, mu, logvar


class GenerativeModel(nn.Module):
    """
    Generative Model encapsulating the AutoEncoder for sequence generation.

    Provides functionality to perform forward passes and calculate the loss for training.

    Args:
        params (Namespace): Configuration parameters containing model dimensions and sequence lengths.

    Attributes:
        autoencoder (AutoEncoder): AutoEncoder module.
        params (Namespace): Model configuration parameters.
    """
    def __init__(self, params):
        super().__init__()
        self.autoencoder = AutoEncoder(params)
        self.params = params

    def forward(self, his_seq, pred_mu, pred_sigma):
        """
        Forward pass for the Generative Model.

        Args:
            his_seq (torch.Tensor): Historical sequence tensor of shape (batch_size, seq_len, c_in).
            pred_mu (torch.Tensor): Prediction mean tensor of shape (batch_size, pred_len, c_in).
            pred_sigma (torch.Tensor): Prediction standard deviation tensor of shape (batch_size, pred_len, c_in).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - Reconstructed sequence of shape (batch_size, seq_len, c_in).
                - Reconstructed trend component of shape (batch_size, pred_len, c_in).
                - Reconstructed seasonal component of shape (batch_size, pred_len, c_in).
                - Mean of the latent distribution of shape (batch_size, latent_dim).
                - Log variance of the latent distribution of shape (batch_size, latent_dim).
        """
        recon_seq, recon_trend, recon_season, mu, logvar = self.autoencoder(his_seq, pred_mu, pred_sigma)
        return recon_seq, recon_trend, recon_season, mu, logvar

    def calculate_loss(self, recon_seq, original_seq, mu, logvar):
        """
        Calculates the loss for the Generative Model.

        Combines Mean Squared Error (MSE) loss for reconstruction and Kullback-Leibler (KL) divergence
        for the latent distribution regularization.

        Args:
            recon_seq (torch.Tensor): Reconstructed sequence tensor of shape (batch_size, seq_len, c_in).
            original_seq (torch.Tensor): Original sequence tensor of shape (batch_size, seq_len, c_in).
            mu (torch.Tensor): Mean of the latent distribution of shape (batch_size, latent_dim).
            logvar (torch.Tensor): Log variance of the latent distribution of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Total loss combining reconstruction and KL divergence.
        """
        pred_len = self.params.pred_len
        original_seq_pred = original_seq[:, -pred_len:, :]
        recon_seq_truncated = recon_seq[:, -pred_len:, :]

        recon_loss = F.mse_loss(recon_seq_truncated, original_seq_pred, reduction='sum')
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * self.params.beta

        total_loss = recon_loss + kl_loss
        return total_loss