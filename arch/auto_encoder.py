import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from torchinfo import summary
from arch.noise import AddNoise

class Encoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 conv_filters: List[int],
                 conv_kernels: List[int],
                 conv_strides: List[int],
                 latent_dim: int,
                 dropout_rate: float = 0.2):  # Added dropout rate parameter
        """
        Encoder network for VAE.
        
        Args:
            input_shape: (height, width, channels)
            conv_filters: List of number of filters for each conv layer
            conv_kernels: List of kernel sizes for each conv layer
            conv_strides: List of strides for each conv layer
            latent_dim: Dimension of latent space
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.num_conv_layers = len(conv_filters)
        self.dropout_rate = dropout_rate
        
        # Build the encoder network
        self.conv_layers = self._build_conv_layers()
        self.flatten = nn.Flatten()
        
        # Calculate flattened size for dense layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            dummy_output = self.conv_layers(dummy_input)
            self.flatten_size = dummy_output.view(1, -1).size(1)
        
        # Mean and variance dense layers for latent space
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_var = nn.Linear(self.flatten_size, latent_dim)
    
    def _build_conv_layers(self) -> nn.Sequential:
        """Build convolutional layers."""
        layers = []
        in_channels = self.input_shape[2]
        
        for f, k, s in zip(self.conv_filters, self.conv_kernels, self.conv_strides):
            padding = (k - 1) // 2
            layers.extend([
                nn.Conv2d(in_channels, f, kernel_size=k, stride=s, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(f),
                nn.Dropout2d(self.dropout_rate)  # Added 2D dropout after each conv block
            ])
            in_channels = f
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(1) != self.input_shape[2]:
            x = x.permute(0, 3, 1, 2)
            
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dropout(x)  # Added dropout before final dense layers
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def summary(self, batch_size: int = 32) -> None:
        """Print model summary."""
        print("\nEncoder Summary:")
        summary(self, input_size=(batch_size, self.input_shape[2], 
                                self.input_shape[0], self.input_shape[1]))


class Decoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 conv_filters: List[int],
                 conv_kernels: List[int],
                 conv_strides: List[int],
                 latent_dim: int,
                 dropout_rate: float = 0.2):  # Added dropout rate parameter
        """
        Decoder network for VAE.
        
        Args:
            input_shape: (height, width, channels)
            conv_filters: List of number of filters for each conv layer
            conv_kernels: List of kernel sizes for each conv layer
            conv_strides: List of strides for each conv layer
            latent_dim: Dimension of latent space
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.num_conv_layers = len(conv_filters)
        self.dropout_rate = dropout_rate
        
        # Build the decoder network
        # First, calculate the initial reshape size
        encoder = Encoder(input_shape, conv_filters, conv_kernels, 
                         conv_strides, latent_dim)
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            dummy_output = encoder.conv_layers(dummy_input)
            self.reshape_size = dummy_output.size()[1:]
        
        self.flatten_size = np.prod(self.reshape_size)
        
        # Build layers
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)  # Added dropout after dense layer
        )
        
        self.decoder_conv = self._build_conv_transpose_layers()
    
    def _build_conv_transpose_layers(self) -> nn.Sequential:
        """Build transpose convolutional layers."""
        layers = []
        filters = self.conv_filters[::-1]  # Reverse the filter list
        kernels = self.conv_kernels[::-1]  # Reverse the kernel list
        strides = self.conv_strides[::-1]  # Reverse the stride list
        
        for i in range(len(filters) - 1):
            padding = (kernels[i] - 1) // 2
            out_padding = strides[i] - 1 if strides[i] > 1 else 0
            
            layers.extend([
                nn.ConvTranspose2d(
                    filters[i], filters[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=padding,
                    output_padding=out_padding
                ),
                nn.ReLU(),
                nn.BatchNorm2d(filters[i + 1]),
                nn.Dropout2d(self.dropout_rate)  # Added 2D dropout after each conv block
            ])
        
        # Final layer to get back to original number of channels
        padding = (kernels[-1] - 1) // 2
        out_padding = strides[-1] - 1 if strides[-1] > 1 else 0
        
        layers.extend([
            nn.ConvTranspose2d(
                filters[-1], self.input_shape[2],
                kernel_size=kernels[-1],
                stride=strides[-1],
                padding=padding,
                output_padding=out_padding
            ),
            nn.Sigmoid()  # No dropout after final sigmoid activation
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_dense(z)
        x = x.view(-1, *self.reshape_size)
        x = self.decoder_conv(x)
        return x

    
    def summary(self, batch_size: int = 32) -> None:
        """Print model summary."""
        print("\nDecoder Summary:")
        summary(self, input_size=(batch_size, self.latent_dim))


class VAE(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (256, 64, 1),  # Default shape
                 conv_filters: List[int] = [64, 128, 256, 512],     # Default filters
                 conv_kernels: List[int] = [3, 3, 3, 3],           # Default kernels
                 conv_strides: List[int] = [2, 2, 2, 2],           # Default strides
                 latent_dim: int = 128,                            # Default latent dim
                 dropout_rate: float = 0.2,
                 noise_factor: float = 0.0,
                 seed: int = 42
                 ):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.seed = seed
        
        # Initialize the noise transform
        self.noise_transform = AddNoise(noise_factor)
        
        # Build encoder and decoder with dropout
        self.encoder = Encoder(
            input_shape, conv_filters, conv_kernels, 
            conv_strides, latent_dim, dropout_rate
        )
        self.decoder = Decoder(
            input_shape, conv_filters, conv_kernels, 
            conv_strides, latent_dim, dropout_rate
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional denoising.
        Returns:
            - reconstruction: reconstructed clean image
            - mu: mean of latent distribution
            - log_var: log variance of latent distribution
            - noisy_x: noisy version of input (if noise_factor > 0)
        """
        # Store original input for loss calculation
        original_x = x
        
        # Apply noise transformation
        noisy_x = self.noise_transform(x)
        
        # If no noise was added, set noisy_x to None for clarity
        if self.noise_transform.noise_factor <= 0.0:
            noisy_x = None
            x_for_encoder = original_x
        else:
            x_for_encoder = noisy_x
            
        # Regular VAE forward pass
        mu, log_var = self.encoder(x_for_encoder)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        
        # Handle permutation if needed
        if original_x.size(1) != self.input_shape[2]:
            reconstruction = reconstruction.permute(0, 2, 3, 1)
            if noisy_x is not None:
                noisy_x = noisy_x.permute(0, 2, 3, 1)
        
        return reconstruction, mu, log_var, noisy_x
    
    def sample(self, num_samples: int = 1, temp: float = 1.0):
        """
        Sample from the latent space and generate audio spectrograms.
        Args:
            num_samples: Number of samples to generate
            temp: Temperature parameter for sampling (higher = more diverse but potentially less stable)
        """
        self.eval()  # Set to evaluation mode
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim, device=device) * temp
            # Decode the latent vectors
            samples = self.decoder(z)
            
        return samples

    def interpolate(self, spec1, spec2, steps: int = 10):
        """
        Interpolate between two spectrograms in latent space.
        Args:
            spec1: First spectrogram
            spec2: Second spectrogram
            steps: Number of interpolation steps
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Encode both spectrograms
            mu1, _ = self.encoder(spec1.to(device))
            mu2, _ = self.encoder(spec2.to(device))
            
            # Create interpolation points
            alphas = torch.linspace(0, 1, steps)
            interpolated = []
            
            # Interpolate in latent space
            for alpha in alphas:
                z = mu1 * (1 - alpha) + mu2 * alpha
                interpolated.append(self.decoder(z))
                
        return torch.stack(interpolated)

    def conditional_sample(self, condition, num_samples: int = 1, temp: float = 1.0):
        """
        Sample from the latent space with a condition (e.g., specific attributes)
        Args:
            condition: Conditioning vector/tensor
            num_samples: Number of samples to generate
            temp: Temperature parameter
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            # Sample random noise
            z = torch.randn(num_samples, self.latent_dim - condition.shape[1], device=device) * temp
            # Concatenate with condition
            z_cond = torch.cat([z, condition.repeat(num_samples, 1)], dim=1)
            # Generate samples
            samples = self.decoder(z_cond)
            
        return samples
    
    def _init_weights(self, m):
        """Initialize model weights with Xavier initialization."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.manual_seed(self.seed)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def summary(self, batch_size: int = 32) -> None:
        """Print model summary."""
        print("\nVAE Summary:")
        self.encoder.summary(batch_size)
        self.decoder.summary(batch_size)
        print(f"\nLatent dimension: {self.latent_dim}")
        print(f"Noise factor: {self.noise_transform.noise_factor}")

    @property
    def noise_factor(self) -> float:
        """Get current noise factor."""
        return self.noise_transform.noise_factor
    
    @noise_factor.setter
    def noise_factor(self, value: float):
        """Set noise factor."""
        self.noise_transform.noise_factor = value