import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization module for normalizing and denormalizing input features."""
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
        """Forward pass for RevIN that either normalizes or denormalizes input."""
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
        """Initializes the learnable affine parameters."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """Computes the mean and standard deviation of the input tensor."""
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """Normalizes the input tensor using computed statistics."""
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
        """Denormalizes the input tensor using stored statistics."""
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
    """Residual block with dropout and residual scaling."""
    def __init__(self, in_features, dropout_rate=0.1, scale=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, in_features)
        )
        self.scale = scale
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        out = self.block(x)
        out = self.dropout(out)
        return self.activation(x + self.scale * out)


class Encoder(nn.Module):
    """Encoder module for transforming input sequences into latent space."""
    def __init__(self, params, revin):
        super().__init__()
        self.revin = revin
        self.params = params

        self.his_dim = params.c_in * params.seq_len
        self.his_proj = nn.Linear(self.his_dim, params.inter_dim // 2)
        self.pred_proj = nn.Linear(params.c_in * 2, params.inter_dim // 2)

        hidden_dim = params.inter_dim

        self.encoder_common = nn.Sequential(
            nn.Linear(params.inter_dim, hidden_dim),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim, dropout_rate=0.2),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim, dropout_rate=0.2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, params.latent_dim * 2)
        )

    def forward(self, his_seq, pred_mu, pred_sigma):
        """Forward pass that encodes input sequences into latent space."""
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
    """Decoder module for transforming latent representations back to sequence space."""
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
            ResidualBlock(hidden_dim, dropout_rate=0.2),
            nn.LeakyReLU(),
            ResidualBlock(hidden_dim, dropout_rate=0.2),
            nn.LeakyReLU()
        )

        self.shared_layer = nn.Sequential(
            ResidualBlock(hidden_dim, dropout_rate=0.2),
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
        """Forward pass that decodes latent vectors into reconstructed sequences."""
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
    """Variational autoencoder for sequence data with reversible instance normalization."""
    def __init__(self, params):
        super().__init__()
        self.revin = RevIN(num_features=params.c_in)
        self.encoder = Encoder(params, self.revin)
        self.decoder = Decoder(params, self.revin)
        self.params = params

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, his_seq, pred_mu, pred_sigma):
        """Forward pass that processes sequences through the autoencoder."""
        his_seq_norm = self.revin(his_seq, mode='norm')
        enc = self.encoder(his_seq_norm, pred_mu, pred_sigma)
        mu, logvar = enc.chunk(2, dim=-1)
        
        hidden_z = self.reparameterize(mu, logvar)
        recon_seq, recon_trend, recon_season = self.decoder(hidden_z, his_seq_norm)
        
        recon_seq = self.revin(recon_seq, mode='denorm')
        return recon_seq, recon_trend, recon_season, mu, logvar


class GenerativeModel(nn.Module):
    """Generative model that wraps an autoencoder for sequence generation."""
    def __init__(self, params):
        super().__init__()
        self.autoencoder = AutoEncoder(params)
        self.params = params

    def forward(self, his_seq, pred_mu, pred_sigma):
        """Forward pass through the generative model."""
        recon_seq, recon_trend, recon_season, mu, logvar = self.autoencoder(his_seq, pred_mu, pred_sigma)
        return recon_seq, recon_trend, recon_season, mu, logvar

    def calculate_loss(self, recon_seq, original_seq, mu, logvar):
        """Calculates reconstruction and KL divergence losses."""
        pred_len = self.params.pred_len
        original_seq_pred = original_seq[:, -pred_len:, :]
        recon_seq_truncated = recon_seq[:, -pred_len:, :]

        recon_loss = F.mse_loss(recon_seq_truncated, original_seq_pred, reduction='sum')
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * self.params.beta

        total_loss = recon_loss + kl_loss
        return total_loss