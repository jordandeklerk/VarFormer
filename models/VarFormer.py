import torch
import torch.nn as nn

from ..layers.Memory_Transformer import *
from ..layers.Generative import *


class Model(nn.Module):
    """Combines Transformer and Generative components for sequence prediction."""
    def __init__(self, params):
        super().__init__()
        self.transformer = TransformerModel(params)
        self.generative_model = GenerativeModel(params)
        self.params = params

    def forward(self, src, src_mark, tgt, tgt_mark):
        """Processes sequences through Transformer and Generative Model to produce reconstructions and latent variables."""
        trans_mu, trans_sigma = self.transformer(src, src_mark, tgt, tgt_mark)
        recon_seq, recon_trend, recon_season, gen_mu, gen_logvar = self.generative_model(src, trans_mu, trans_sigma)
        return recon_seq, recon_trend, recon_season, trans_mu, trans_sigma, gen_mu, gen_logvar

    def calculate_loss(self, recon_seq, recon_trend, recon_season, trans_mu, trans_sigma, gen_mu, gen_logvar, original_seq):
        """Calculates total loss combining NLL, reconstruction loss, and KL divergence."""
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
        """Generates forecasts using the Model without calculating loss."""
        trans_mu, trans_sigma = self.transformer.forecast(src, src_mark, tgt, tgt_mark)
        recon_seq, recon_trend, recon_season, gen_mu, gen_logvar = self.generative_model(src, trans_mu, trans_sigma)
        return recon_seq, recon_trend, recon_season, trans_mu, trans_sigma, gen_mu, gen_logvar