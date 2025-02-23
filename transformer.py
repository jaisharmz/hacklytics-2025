import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np 
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

class TransformerAutoencoder(nn.Module):
    def __init__(self, time_steps, d_model=128, nhead=4, num_layers=3, latent_dim=64, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # Project feature dim (1) to d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent Space Projection (Global Pooling to get a single vector)
        self.latent_proj = nn.Linear(d_model, latent_dim)  # Reduce d_model → latent_dim
        self.latent_norm = nn.LayerNorm(latent_dim)  # Normalize latent space

        # Reverse Projection for Decoder Input
        self.mult_latent = 10
        self.reverse_proj = nn.Linear(latent_dim, d_model)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output Projection
        self.output_proj = nn.Linear(d_model, 1)

        # Normalization layers
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [batch_size, num_features=1, time_steps]
        """
        # Prepare input: [batch, time_steps, features]
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # Project input to d_model
        x = self.pos_encoder(x)

        # Encoder Forward Pass
        enc_output = self.encoder(x)  # [batch, time_steps, d_model]
        enc_output = self.encoder_norm(enc_output)

        # Global Mean Pooling to obtain a latent vector
        # latent = enc_output.mean(dim=1, keepdim=True)  # [batch, 1, d_model] # TODO: instead of taking the mean across the time dimension, take the first 10 entries of the time dimension, and then reshape after the next latent_proj to get [batch, 1, latent_dim * 10]
        if enc_output.shape[1] < self.mult_latent:
          enc_output = enc_output.repeat(1, math.ceil(self.mult_latent / enc_output.shape[1]), 1)
          enc_output = enc_output[:,:self.mult_latent, :]
        latent = enc_output[:,:self.mult_latent,:]
        # Project to latent space and normalize → this is the true latent representation
        latent_code = self.latent_proj(latent)       # [batch, 10, latent_dim]
        latent_code = self.latent_norm(latent_code)
        latent_result = latent_code.view(latent_code.shape[0], -1)

        # Prepare input for the decoder: use reverse projection on the latent_code
        decoder_input = self.reverse_proj(latent_code)  # [batch, 10, d_model]

        # Expand decoder input back to sequence length
        repeated_decoder_input = decoder_input.repeat(1, math.ceil(x.shape[1] / self.mult_latent), 1)
        repeated_decoder_input = repeated_decoder_input[:,:x.shape[1],:] # [batch, time_steps, d_model]
        # Break symmetry: apply positional encoding to the repeated latent representation
        repeated_decoder_input = self.pos_encoder(repeated_decoder_input)

        # Decoder Forward Pass
        # Instead of using enc_output as memory (which comes before the bottleneck), we now use
        # repeated_decoder_input for both target and memory so the decoder solely relies on the bottleneck.
        dec_output = self.decoder(repeated_decoder_input, repeated_decoder_input)  # [batch, time_steps, d_model]
        dec_output = self.decoder_norm(dec_output)

        # Output Projection
        out = self.output_proj(dec_output)  # [batch, time_steps, 1]
        out = out.permute(0, 2, 1)  # Convert back to [batch, 1, time_steps]

        # Return both the reconstructed output and the latent representation (squeezed to remove the time dim)
        return out, latent_result

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device, end="\n\n\n")
model = TransformerAutoencoder(1257)
model.load_state_dict(torch.load("/Users/christophersun/Documents/hacklytics-2025/model_new.pth", map_location=torch.device("cpu")))
model = model.to(device)
model.eval()

def find_arbitrage(X, model, device, trend_of_interest, interval_min, interval_max, top_k):
  assert interval_min < interval_max
  # Looks at the past month's worth of data and finds stocks whose temporal embeddings are the most similar
  # but whose stock data have the largest discrepancy in the past month

  # Embed data outside interval and compare similarity
  interpolation = np.linspace(X[:,interval_min], X[:,interval_max], num=interval_max - interval_min, axis=1)
  X_outside = X.copy()
  X_outside[:,interval_min:interval_max] = interpolation
  model.eval()
  dataset = TensorDataset(torch.tensor(X_outside, dtype=torch.float32).unsqueeze(1))
  loader = DataLoader(dataset, batch_size=1, shuffle=False)
  latents_lst = []
  for (batch,) in loader:
    _, latents = model(batch.to(device))
    latents = latents.cpu().detach().numpy()
    latents_lst.append(latents)
  latents_lst = np.concatenate(latents_lst)
  similarity = np.sum((latents_lst[trend_of_interest] - latents_lst) ** 2, axis=1)

  # Embed data within interval and compare dissimilarity
  X_within = X[:,interval_min:interval_max]
  model.eval()
  dataset = TensorDataset(torch.tensor(X_within, dtype=torch.float32).unsqueeze(1))
  loader = DataLoader(dataset, batch_size=1, shuffle=False)
  latents_lst = []
  for (batch,) in loader:
    _, latents = model(batch.to(device))
    latents = latents.cpu().detach().numpy()
    latents_lst.append(latents)
  latents_lst = np.concatenate(latents_lst)
  dissimilarity = -1 * np.sum((latents_lst[trend_of_interest] - latents_lst) ** 2, axis=1)

  cost = similarity * 0.95 + dissimilarity * 0.05
  return cost

spy500 = pd.read_csv("/Users/christophersun/Documents/hacklytics-2025/SPY_500_5y.csv")
# columns = spy500.iloc[0].values[1:]
# spy500 = spy500.iloc[2:]
# spy500.columns = ["Price"] + columns.tolist()
# spy500 = spy500.dropna(axis=1)
X = spy500.iloc[2:,1:].dropna(axis=1).T.astype(float)
X = X / X.iloc[:,0].values.reshape(488, 1)
X = X.values

print("starting arbitrage")
costs = find_arbitrage(X, model, device, 0, 900, 1256, top_k=None)
print("finished arbitrage")