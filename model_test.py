import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

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

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
spy500 = pd.read_csv("SPY_500_5y.csv")
y = spy500.iloc[0,1:]
y = y[np.logical_not(spy500.iloc[2:,1:].isna().any(axis=0))].tolist()
X = spy500.iloc[2:,1:].dropna(axis=1).T.astype(float)
X = X / X.iloc[:,0].values.reshape(488, 1)
X_t, X_v, y_t, y_v = train_test_split(X.values, y, test_size=0.3, random_state=0)
X_t = X_t.reshape(X_t.shape[0], 1, X_t.shape[-1])
X_v = X_v.reshape(X_v.shape[0], 1, X_v.shape[-1])
model = TransformerAutoencoder(X_t.shape[2]).to(device)
model.load_state_dict(torch.load("model_new.pth", map_location=device))
train_dataset = TensorDataset(torch.tensor(X_t, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_v, dtype=torch.float32))
all_dataset = TensorDataset(torch.tensor(X.values.reshape(X.shape[0], 1, X.shape[-1]), dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
all_loader = DataLoader(all_dataset, batch_size=32, shuffle=False)

# model.eval()
# latents = []
# avg_daily_percent_change = []
# overall_percent_change = []
# train_val = [0 for i in range(len(train_loader.dataset))] + [1 for i in range(len(val_loader.dataset))]

# special_idx = 1
# ref = torch.tensor(X_t[special_idx], dtype=torch.float32).unsqueeze(0).to(device)
# ref = ref.view(ref.shape[0],-1)
# ref_mean = ref.mean()
# ref_std = ref.std()
# corrs = []

# # with torch.no_grad():
# #     for (batch, ) in all_loader:
# #         x = batch.to(device)
# #         output, latent = model(x)
# #         # print(latent.shape)
# #         latents.append(latent.cpu().detach().numpy())
# #         avg_daily_percent_change += (x.diff() / x[:,:,:-1]).mean(dim=-1).flatten().tolist()
# #         overall_percent_change += ((x[:,:,-1] - x[:,:,0]) / x[:,:,0]).flatten().tolist()
# with torch.no_grad():
#     for (batch, ) in train_loader:
#         x = batch.to(device)
#         output, latent = model(x)
#         # print(latent.shape)
#         latents.append(latent.cpu().detach().numpy())
#         avg_daily_percent_change += (x.diff() / x[:,:,:-1]).mean(dim=-1).flatten().tolist()
#         overall_percent_change += ((x[:,:,-1] - x[:,:,0]) / x[:,:,0]).flatten().tolist()
#         others = x.view(x.shape[0], -1)
#         others_mean = others.mean(dim=1, keepdim=True)
#         others_std = others.std(dim=1, keepdim=True)
#         cov = ((others - others_mean) * (ref - ref_mean)).sum(dim=1) / (others.shape[-1] - 1)
#         corr = cov / (ref_std * others_std.squeeze())
#         corrs += corr.tolist()

# with torch.no_grad():
#     for (batch, ) in val_loader:
#         x = batch.to(device)
#         output, latent = model(x)
#         # print(latent.shape)
#         latents.append(latent.cpu().detach().numpy())
#         avg_daily_percent_change += (x.diff() / x[:,:,:-1]).mean(dim=-1).flatten().tolist()
#         overall_percent_change += ((x[:,:,-1] - x[:,:,0]) / x[:,:,0]).flatten().tolist()
#         others = x.view(x.shape[0], -1)
#         others_mean = others.mean(dim=1, keepdim=True)
#         others_std = others.std(dim=1, keepdim=True)
#         cov = ((others - others_mean) * (ref - ref_mean)).sum(dim=1) / (others.shape[-1] - 1)
#         corr = cov / (ref_std * others_std.squeeze())
#         corrs += corr.tolist()
# latents = np.concatenate(latents)
# overall_percent_change_clipped = np.clip(overall_percent_change, np.percentile(overall_percent_change, 5), np.percentile(overall_percent_change, 95))
# tickers = pd.Series(y_t + y_v)
# X_total = np.concatenate((X_t, X_v))

price_data_dict = json.load(open("price_data_hacklytics.json", "r"))
all_price_data = {key: np.array(value)[:,1] for key, value in price_data_dict.items()}
# SEQ_LEN = 1257
all_outputs = []
all_latents = []
for key, value in all_price_data.items():
  price_seq = torch.tensor(value, dtype=torch.float32).to(device).view(1,1,-1)
  price_seq = price_seq / price_seq[:,:,0]
  outputs, latent = model(price_seq)
  all_outputs.append(outputs.cpu().detach().numpy())
  all_latents.append(latent.cpu().detach().numpy())
cosines = []
for i, (key, value) in enumerate(all_price_data.items()):
  cosines.append(nn.CosineSimilarity()(torch.tensor(all_outputs[i]).squeeze().view(1,-1), torch.tensor(value).view(1,-1)).item())

# CHRISTOPHER SUN FUNCTIONS
def find_similar_within_interval(X, model, device, trend_of_interest, interval_min, interval_max, top_k):
  # trend_of_interest: index of user selected trend in the data
  # interval_min and interval_max determined by user (interpolate to [0, sequence length])
  assert interval_min < interval_max
  X_subset = X[:,:,interval_min:interval_max]
  model.eval()
  _, latents = model(torch.tensor(X_subset, dtype=torch.float32).to(device))
  latents = latents.cpu().detach().numpy()
  closest_ranks = np.sum((all_latents[trend_of_interest] - all_latents) ** 2, axis=1).argsort()
  closest_ranks = closest_ranks[:top_k]
  return {closest_ranks[i]: X[i] for i in range(closest_ranks.shape[0])}

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

costs = find_arbitrage(X.values, model, device, 6, 900, 1256, None)
print(costs.argsort())