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