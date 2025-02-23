import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return x + self.pe[:, :x.size(1), :]

class TransformerAutoencoder(nn.Module):
    def __init__(self, time_steps, d_model=128, nhead=4, num_layers=3, latent_dim=64, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # Project feature dim (1) to d_model
        self.pos_encoder = PositionalEncoding(d_model, max_len=time_steps)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, 
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent Space Projection
        self.latent_proj = nn.Linear(d_model, latent_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output Projection
        self.output_proj = nn.Linear(d_model, 1)

        # Residual Layer Norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [batch_size, num_features=1, time_steps]
        x = x.permute(0, 2, 1)  # [batch_size, time_steps, num_features]
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Encoder Forward Pass
        enc_output = self.encoder(x)
        enc_output = self.layer_norm(enc_output)

        # Latent Space Representation
        latent = self.latent_proj(enc_output)

        # Decoder Forward Pass (Reconstruction)
        dec_output = self.decoder(enc_output, enc_output)  # Using encoded input as target for reconstruction
        dec_output = self.layer_norm(dec_output)

        # Output Projection
        out = self.output_proj(dec_output)  # Shape: [batch_size, time_steps, 1]
        out = out.permute(0, 2, 1)  # Back to [batch_size, 1, time_steps]
        
        return out, latent  # Returning both the reconstruction and latent space

# Example Usage:
batch_size = 32
time_steps = 100  # Length of time series
x = torch.randn(batch_size, 1, time_steps)  # Example batch

model = TransformerAutoencoder(time_steps)
output, latent = model(x)
print(output.shape, latent.shape)  # Expected: [32, 1, 100] and [32, 100, 64]

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