import torch
import torch.nn as nn

from layers.Memory_Transformer import *
from layers.Generative import *


class Model(nn.Module):
    """
    Comprehensive Model integrating Transformer and Generative components for sequence prediction.

    This model combines a Transformer-based module for capturing temporal dependencies and a
    Generative Model for reconstructing and generating sequences. It provides functionality for 
    forward passes, loss calculation, and forecasting.

    Args:
        params (Namespace): Configuration parameters containing model dimensions, sequence lengths,
                            and hyperparameters required for the Transformer and Generative Model.

    Attributes:
        transformer (TransformerModel): Transformer-based module for encoding input sequences.
        generative_model (GenerativeModel): Generative module for reconstructing and generating sequences.
        params (Namespace): Model configuration parameters.
    """
    def __init__(self, params):
        super().__init__()
        self.transformer = TransformerModel(params)
        self.generative_model = GenerativeModel(params)
        self.params = params

    def forward(self, src, src_mark, tgt, tgt_mark):
        """
        Forward pass through the Model.

        Processes the source and target sequences through the Transformer and Generative Model
        to produce reconstructed sequences and latent variables.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_seq_len, feature_dim).
            src_mark (torch.Tensor): Source sequence markers or positional encodings.
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, feature_dim).
            tgt_mark (torch.Tensor): Target sequence markers or positional encodings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - recon_seq (torch.Tensor): Reconstructed sequence of shape (batch_size, seq_len, feature_dim).
                - recon_trend (torch.Tensor): Reconstructed trend component of shape (batch_size, pred_len, feature_dim).
                - recon_season (torch.Tensor): Reconstructed seasonal component of shape (batch_size, pred_len, feature_dim).
                - trans_mu (torch.Tensor): Transformer output mean of shape (batch_size, pred_len, feature_dim).
                - trans_sigma (torch.Tensor): Transformer output standard deviation of shape (batch_size, pred_len, feature_dim).
                - gen_mu (torch.Tensor): Generative model latent mean of shape (batch_size, latent_dim).
                - gen_logvar (torch.Tensor): Generative model latent log variance of shape (batch_size, latent_dim).
        """
        trans_mu, trans_sigma = self.transformer(src, src_mark, tgt, tgt_mark)
        recon_seq, recon_trend, recon_season, gen_mu, gen_logvar = self.generative_model(src, trans_mu, trans_sigma)
        return recon_seq, recon_trend, recon_season, trans_mu, trans_sigma, gen_mu, gen_logvar

    def calculate_loss(self, recon_seq, recon_trend, recon_season, trans_mu, trans_sigma, gen_mu, gen_logvar, original_seq):
        """
        Calculates the total loss for the Model.

        Combines Negative Log-Likelihood (NLL) loss from the Transformer outputs and
        the reconstruction loss along with Kullback-Leibler (KL) divergence from the Generative Model.

        Args:
            recon_seq (torch.Tensor): Reconstructed sequence tensor of shape (batch_size, seq_len, feature_dim).
            recon_trend (torch.Tensor): Reconstructed trend component of shape (batch_size, pred_len, feature_dim).
            recon_season (torch.Tensor): Reconstructed seasonal component of shape (batch_size, pred_len, feature_dim).
            trans_mu (torch.Tensor): Transformer output mean of shape (batch_size, pred_len, feature_dim).
            trans_sigma (torch.Tensor): Transformer output standard deviation of shape (batch_size, pred_len, feature_dim).
            gen_mu (torch.Tensor): Generative model latent mean of shape (batch_size, latent_dim).
            gen_logvar (torch.Tensor): Generative model latent log variance of shape (batch_size, latent_dim).
            original_seq (torch.Tensor): Original sequence tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Total loss combining NLL, reconstruction loss, and KL divergence.
        """
        pred_len = self.params.pred_len
        original_seq_pred = original_seq[:, -pred_len:, :]

        # Negative Log-Likelihood (NLL) Loss
        nll = torch.log(trans_sigma[:, -pred_len:, :]) + \
              0.5 * ((original_seq_pred - trans_mu[:, -pred_len:, :]) / trans_sigma[:, -pred_len:, :]).pow(2)
        trans_loss = -torch.mean(nll) * self.params.alpha

        # Reconstruction loss and KL divergence from the Generative Model
        gen_loss = self.generative_model.calculate_loss(recon_seq, original_seq, gen_mu, gen_logvar)

        # Total loss is the sum of Transformer loss and Generative Model loss
        total_loss = trans_loss + gen_loss
        return total_loss

    def forecast(self, src, src_mark, tgt, tgt_mark):
        """
        Generates forecasts using the Model.

        Processes the source and target sequences through the Transformer and Generative Model
        to produce forecasted sequences and latent variables without calculating the loss.

        Args:
            src (torch.Tensor): Source sequence tensor of shape (batch_size, src_seq_len, feature_dim).
            src_mark (torch.Tensor): Source sequence markers or positional encodings.
            tgt (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, feature_dim).
            tgt_mark (torch.Tensor): Target sequence markers or positional encodings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - recon_seq (torch.Tensor): Reconstructed sequence of shape (batch_size, seq_len, feature_dim).
                - recon_trend (torch.Tensor): Reconstructed trend component of shape (batch_size, pred_len, feature_dim).
                - recon_season (torch.Tensor): Reconstructed seasonal component of shape (batch_size, pred_len, feature_dim).
                - trans_mu (torch.Tensor): Transformer output mean of shape (batch_size, pred_len, feature_dim).
                - trans_sigma (torch.Tensor): Transformer output standard deviation of shape (batch_size, pred_len, feature_dim).
                - gen_mu (torch.Tensor): Generative model latent mean of shape (batch_size, latent_dim).
                - gen_logvar (torch.Tensor): Generative model latent log variance of shape (batch_size, latent_dim).
        """
        trans_mu, trans_sigma = self.transformer.forecast(src, src_mark, tgt, tgt_mark)
        recon_seq, recon_trend, recon_season, gen_mu, gen_logvar = self.generative_model(src, trans_mu, trans_sigma)
        return recon_seq, recon_trend, recon_season, trans_mu, trans_sigma, gen_mu, gen_logvar