import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 conv_filters: List[int],
                 conv_kernels: List[int],
                 conv_strides: List[int],
                 latent_dim: int):
        """
        Encoder network for VAE.
        
        Args:
            input_shape: (height, width, channels)
            conv_filters: List of number of filters for each conv layer
            conv_kernels: List of kernel sizes for each conv layer
            conv_strides: List of strides for each conv layer
            latent_dim: Dimension of latent space
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.num_conv_layers = len(conv_filters)
        
        # Build the encoder network
        self.conv_layers = self._build_conv_layers()
        self.flatten = nn.Flatten()
        
        # Calculate flattened size for dense layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            dummy_output = self.conv_layers(dummy_input)
            self.flatten_size = dummy_output.view(1, -1).size(1)
        
        # Mean and variance dense layers for latent space
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
                nn.BatchNorm2d(f)
            ])
            in_channels = f
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            mu: Mean vector of latent space
            log_var: Log variance vector of latent space
        """
        # Ensure input is in correct format (BCHW)
        if x.size(1) != self.input_shape[2]:
            x = x.permute(0, 3, 1, 2)
            
        x = self.conv_layers(x)
        x = self.flatten(x)
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
                 latent_dim: int):
        """
        Decoder network for VAE.
        
        Args:
            input_shape: (height, width, channels)
            conv_filters: List of number of filters for each conv layer
            conv_kernels: List of kernel sizes for each conv layer
            conv_strides: List of strides for each conv layer
            latent_dim: Dimension of latent space
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim
        self.num_conv_layers = len(conv_filters)
        
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
        self.decoder_dense = nn.Linear(latent_dim, self.flatten_size)
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
                nn.BatchNorm2d(filters[i + 1])
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
            nn.Sigmoid()
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed image tensor of shape (batch_size, channels, height, width)
        """
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
                 input_shape: Tuple[int, int, int],
                 conv_filters: List[int],
                 conv_kernels: List[int],
                 conv_strides: List[int],
                 latent_dim: int):
        """
        Variational Autoencoder.
        
        Args:
            input_shape: (height, width, channels)
            conv_filters: List of number of filters for each conv layer
            conv_kernels: List of kernel sizes for each conv layer
            conv_strides: List of strides for each conv layer
            latent_dim: Dimension of latent space
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Build encoder and decoder
        self.encoder = Encoder(input_shape, conv_filters, conv_kernels, 
                             conv_strides, latent_dim)
        self.decoder = Decoder(input_shape, conv_filters, conv_kernels, 
                             conv_strides, latent_dim)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, channels)
            
        Returns:
            reconstruction: Reconstructed input with same shape as input
            mu: Mean of latent space
            log_var: Log variance of latent space
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        
        # Convert reconstruction from BCHW to BHWC if input was BHWC
        if x.size(1) != self.input_shape[2]:  # If input was BHWC
            reconstruction = reconstruction.permute(0, 2, 3, 1)
            
        return reconstruction, mu, log_var
    
    def summary(self, batch_size: int = 32) -> None:
        """Print model summary."""
        print("\nVAE Summary:")
        self.encoder.summary(batch_size)
        self.decoder.summary(batch_size)
        print(f"\nLatent dimension: {self.latent_dim}")


if __name__ == "__main__":
    # Example usage
    vae = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_dim=2
    )
    
    # Print model summary
    vae.summary()
    
    # Test forward pass
    x = torch.randn(32, 28, 28, 1)  # Example batch of 32 images
    recon, mu, log_var = vae(x)
    print("\nOutput shapes:")
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mean shape: {mu.shape}")
    print(f"Latent log variance shape: {log_var.shape}")