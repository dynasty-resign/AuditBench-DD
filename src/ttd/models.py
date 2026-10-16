from __future__ import annotations

import math

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(math.gcd(8, cout), cout),
            nn.SiLU(),
        )


class LatentAutoencoder(nn.Module):
    """Compact autoencoder appropriate for 32/64 px datasets.

    Stable-Diffusion's 8x downsampling leaves CIFAR with a 4x4 latent and is a poor
    default. This backbone uses 4x downsampling and can be replaced by a checkpoint.
    """

    def __init__(self, latent_channels: int = 8, width: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, width),
            ConvBlock(width, width, 2),
            ConvBlock(width, width * 2, 2),
            nn.Conv2d(width * 2, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            ConvBlock(latent_channels, width * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(width * 2, width),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, 3, 3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class LatentFeatureEncoder(nn.Module):
    def __init__(self, latent_channels: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(latent_channels, embed_dim),
            ConvBlock(embed_dim, embed_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(z), dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        emb = t.float()[:, None] * freq[None]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return self.proj(emb)


class NoisePredictor(nn.Module):
    def __init__(self, channels: int, width: int, num_classes: int):
        super().__init__()
        self.time = TimeEmbedding(width)
        self.null_label = num_classes
        self.label = nn.Embedding(num_classes + 1, width)
        self.input = ConvBlock(channels, width)
        self.mid = nn.Sequential(ConvBlock(width, width), ConvBlock(width, width))
        self.output = nn.Conv2d(width, channels, 3, padding=1)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.time(t) + self.label(labels)
        h = self.input(zt) + cond[:, :, None, None]
        return self.output(self.mid(h))


class LatentDiffusion(nn.Module):
    def __init__(self, latent_channels: int, width: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.autoencoder = LatentAutoencoder(latent_channels, width)
        self.features = LatentFeatureEncoder(latent_channels, embed_dim)
        self.denoiser = NoisePredictor(latent_channels, width, num_classes)
        self.register_buffer("latent_mean", torch.zeros(1, latent_channels, 1, 1))
        self.register_buffer("latent_std", torch.ones(1, latent_channels, 1, 1))
        self.register_buffer("latent_stats_fitted", torch.tensor(False))

    def unconditional_labels(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch,), self.denoiser.null_label, dtype=torch.long, device=device)

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != self.latent_mean.shape[1] or std.numel() != self.latent_std.shape[1]:
            raise ValueError("latent statistics do not match latent channel count")
        self.latent_mean.copy_(mean.reshape_as(self.latent_mean))
        self.latent_std.copy_(std.reshape_as(self.latent_std).clamp_min(1e-6))
        self.latent_stats_fitted.fill_(True)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        if not bool(self.latent_stats_fitted):
            raise RuntimeError("latent statistics must be fitted before normalized encoding")
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if not bool(self.latent_stats_fitted):
            raise RuntimeError("latent statistics must be fitted before decoding normalized latents")
        return z * self.latent_std + self.latent_mean

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.encode_raw(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(self.denormalize(z))

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        return self.features(z)


class ConvNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 128):
        super().__init__()
        layers = []
        channels = 3
        for _ in range(3):
            layers += [ConvBlock(channels, width), nn.AvgPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TinyViT(nn.Module):
    def __init__(self, num_classes: int, image_size: int, dim: int = 192, depth: int = 6):
        super().__init__()
        patch = 4
        n = (image_size // patch) ** 2
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 3, dim * 4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(len(x), -1, -1)
        x = self.encoder(torch.cat([cls, x], dim=1) + self.pos)
        return self.head(x[:, 0])


def classifier(name: str, num_classes: int, image_size: int) -> nn.Module:
    if name == "convnet":
        return ConvNet(num_classes)
    if name == "vit_tiny":
        return TinyViT(num_classes, image_size)
    if name == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(num_classes=num_classes)
        if image_size <= 64:
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        return model
    if name == "vgg11":
        from torchvision.models import vgg11

        return vgg11(num_classes=num_classes)
    if name == "mobilenet_v2":
        from torchvision.models import mobilenet_v2

        model = mobilenet_v2(num_classes=num_classes)
        if image_size <= 64:
            model.features[0][0] = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        return model
    raise ValueError(f"unknown architecture: {name}")
