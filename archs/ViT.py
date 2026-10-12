from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np
from datasets import transforms
from scipy.stats import wasserstein_distance
import pickle
import os
import glob
import random

from archs.transformer_layer import ViTBackbone


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class FeatureEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(FeatureEmbedding, self).__init__()
        # Conv + DW-PW + DW-PW
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels // 4, [1, 3, 3], stride=1, padding=[0, 1, 1]
            ),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(
                out_channels // 4,
                out_channels // 2,
                [3, 3, 3],
                stride=1,
                padding=1,
                groups=out_channels // 4,
            ),
            nn.Conv3d(
                out_channels // 2, out_channels // 2, [1, 1, 1], [1, 1, 1], [0, 0, 0]
            ),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(
                out_channels // 2,
                out_channels,
                [3, 3, 3],
                stride=1,
                padding=1,
                groups=out_channels // 2,
            ),
            nn.Conv3d(out_channels, out_channels, [1, 1, 1], [1, 1, 1], [0, 0, 0]),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.apply(self.init_weights)

    @torch.no_grad()
    def init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x : [B, C, T, H, W] = [B, 3, 160, 128, 128]
        stem_feat : [B, dim, 160, 32, 32]
        """
        # print(f'x_before_stem: {x.shape}')
        stem_feat = self.conv(x)
        # print(f'stem_feat: {stem_feat.shape}')
        return stem_feat


class Predictor(nn.Module):
    """
    Extract signal from feature map, feature map [B, C, T, H, W] = [B, 64, 40, 4, 4]
    1. 3D Upsample -> [B, 64, 160, 4, 4]
    2. frame squeeze * 2 -> [B, 64, 160]
    3. channel squeeze -> [B, 1, 160] -> [B, 160]
    """

    def __init__(self, out_channels):
        super(Predictor, self).__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(
                out_channels, out_channels, [3, 1, 1], stride=1, padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ELU(),
            nn.Upsample(scale_factor=(2, 1, 1)),
            nn.Conv3d(
                out_channels, out_channels // 2, [3, 1, 1], stride=1, padding=(1, 0, 0)
            ),
            nn.BatchNorm3d(out_channels // 2),
            nn.ELU(),
        )

        self.channel_squeeze = nn.Conv1d(out_channels // 2, 1, 1, stride=1, padding=0)

    def forward(self, x):
        x_upsample = self.upsample(x)
        x_frame_squeeze = torch.mean(x_upsample, 4)
        x_frame_squeeze = torch.mean(x_frame_squeeze, 3)  # [B, 64, 160]
        x_channel_squeeze = self.channel_squeeze(x_frame_squeeze)  # [B, 1, 160]
        rPPG = x_channel_squeeze.squeeze(1)  # [B, 160]
        return rPPG


class Adapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=None, dropout=0.1):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = dim // 4

        self.adapter = nn.Sequential(
            nn.Linear(dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )

        # Keep the bottleneck projection trainable from the first update while
        # starting with an identity residual path.
        nn.init.kaiming_uniform_(self.adapter[0].weight, a=5**0.5)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.normal_(self.adapter[3].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.adapter[3].bias)

    def forward(self, x):
        return x + self.adapter(x)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Linear layers for mean and variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Store distribution statistics after training
        self.saved_distributions = {}

        # Normalization parameters for Wasserstein distance calculation
        self.normalize_wasserstein = False
        self.wasserstein_norm_method = "l2"

    def encode(self, x):
        """Encode"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode"""
        return self.decoder(z)

    def forward(self, x):
        """Forward propagation"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    def save_distribution(self, x, domain_name="default"):
        """Save distribution statistics of current domain"""
        with torch.no_grad():
            # Flatten input features to [B*S, D]
            x_flat = x.reshape(-1, x.size(-1))
            mu, log_var = self.encode(x_flat)

            # L2 normalize latent mean vectors to unit hypersphere
            normalized_mu = self.normalize_distribution(mu.cpu().numpy(), "unit_sphere")

            # Save normalized latent mean vectors as fingerprint
            self.saved_distributions[domain_name] = {
                "normalized_mu": normalized_mu  # [B*S, d_lat]
            }

    def compute_distribution_distance(self, x, domain_name="default"):
        """Compute Sliced Wasserstein distance between new input and saved distribution"""
        if domain_name not in self.saved_distributions:
            return float("inf")

        with torch.no_grad():
            # Flatten input features to [B*S, D]
            x_flat = x.reshape(-1, x.size(-1))
            mu_new, _ = self.encode(x_flat)
            normalized_mu_new = self.normalize_distribution(
                mu_new.cpu().numpy(), "unit_sphere"
            )

            normalized_mu_saved = self.saved_distributions[domain_name]["normalized_mu"]

            # Compute Sliced Wasserstein Distance
            distance = self.sliced_wasserstein_distance(
                normalized_mu_saved, normalized_mu_new
            )

            return distance

    def sliced_wasserstein_distance(self, X, Y, K=16):
        """Compute Sliced Wasserstein Distance (SWD)

        Args:
            X, Y: np.ndarray of shape [N, D]
            K: number of projection directions

        Returns:
            float: SWD distance
        """
        import numpy as np

        N_X, D = X.shape
        N_Y = Y.shape[0]
        N_hat = min(N_X, N_Y)

        swd = 0.0
        for _ in range(K):
            # Randomly sample unit vectors
            v = np.random.randn(D)
            v = v / (np.linalg.norm(v) + 1e-8)

            # Projection
            proj_X = X @ v
            proj_Y = Y @ v

            # Sort and take first N_hat
            proj_X_sorted = np.sort(proj_X)[:N_hat]
            proj_Y_sorted = np.sort(proj_Y)[:N_hat]

            # Compute 1D Wasserstein-1 distance
            swd += np.mean(np.abs(proj_X_sorted - proj_Y_sorted))

        return swd / K

    def set_wasserstein_params(self, normalize=False, norm_method="l2"):
        """Set normalization parameters for Wasserstein distance calculation"""
        self.normalize_wasserstein = normalize
        self.wasserstein_norm_method = norm_method

    def normalize_distribution(self, data, method="l2"):
        """Normalize distribution data

        Args:
            data (np.ndarray): Input data
            method (str): Normalization method
                - 'l2': L2 normalization
                - 'unit_sphere': Project to unit sphere
                - 'z_score': Standardization (zero mean unit variance)
                - 'minmax': Min-max scale to [0,1]

        Returns:
            np.ndarray: Normalized data
        """
        if len(data) == 0:
            return data

        if method == "l2":
            # L2 normalization
            norm = np.linalg.norm(data)
            return data / (norm + 1e-8) if norm > 1e-8 else data

        elif method == "unit_sphere":
            # Project to unit sphere (L2 norm of each sample is 1)
            norms = np.linalg.norm(data.reshape(len(data), -1), axis=1, keepdims=True)
            return data / (norms + 1e-8)

        elif method == "z_score":
            # Standardization (zero mean, unit variance)
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / (std + 1e-8)

        elif method == "minmax":
            # Min-max scale to [0,1]
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val + 1e-8)

        else:
            return data


class AdapterList(nn.Module):
    def __init__(
        self,
        dim,
        bottleneck_dim=None,
        dropout=0.1,
        initial_adapters=0,
        vae_latent_dim=64,
        vae_hidden_dim=256,
    ):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = (
            bottleneck_dim if bottleneck_dim is not None else min(dim, 32)
        )
        self.dropout = dropout
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dim = vae_hidden_dim
        self.adapters = nn.ModuleList()

        # Separated weight storage
        self.adapter_weights = nn.ParameterList()

        # Store corresponding VAE for each adapter
        self.adapter_vaes = nn.ModuleList()

        # Store input features separately for each adapter
        self.adapter_input_features = []

        # Freeze status flag
        self.adapters_frozen = False

        # Initialize specified number of adapters
        for _ in range(initial_adapters):
            self.add_adapter()

    def add_adapter(self):
        """Dynamically add a new adapter"""
        new_adapter = Adapter(self.dim, self.bottleneck_dim, self.dropout)

        # Ensure new adapter is on correct device
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # If no parameters, use CPU as default device
            device = torch.device("cpu")

        new_adapter = new_adapter.to(device)

        self.adapters.append(new_adapter)

        # Add corresponding routing column vector for new adapter (corresponding to a column of W_l^{mix} in the literature)
        # Initialize to small random values
        new_weight_vec = nn.Parameter(torch.randn(self.dim, device=device) * 1e-3)
        self.adapter_weights.append(new_weight_vec)

        # Add corresponding VAE for new adapter
        new_vae = VAE(self.dim, self.vae_latent_dim, self.vae_hidden_dim)
        new_vae = new_vae.to(device)
        self.adapter_vaes.append(new_vae)

        # Add input feature storage location for new adapter
        self.adapter_input_features.append(None)

        return len(self.adapters) - 1

    def freeze_adapters(self):
        """Freeze all existing adapters"""
        self.adapters_frozen = True
        for adapter in self.adapters:
            for param in adapter.parameters():
                param.requires_grad = False
        for weight in self.adapter_weights:
            weight.requires_grad = False
        for vae in self.adapter_vaes:
            for param in vae.parameters():
                param.requires_grad = False

    def unfreeze_latest_adapter(self):
        """Unfreeze the latest added adapter"""
        if len(self.adapters) > 0:
            for param in self.adapters[-1].parameters():
                param.requires_grad = True
            self.adapter_weights[-1].requires_grad = True
            for param in self.adapter_vaes[-1].parameters():
                param.requires_grad = True

    def save_vae_distributions(self, x, domain_name="default"):
        """Save distribution information for all VAEs"""
        for vae in self.adapter_vaes:
            vae.save_distribution(x, domain_name)

    def compute_vae_distances(self, x, domain_name="default"):
        """Compute distribution distance of input on all VAEs"""
        distances = []
        for vae in self.adapter_vaes:
            distance = vae.compute_distribution_distance(x, domain_name)
            distances.append(distance)
        return distances

    def set_wasserstein_params(self, normalize=False, norm_method="l2"):
        """Set normalization parameters for Wasserstein distance calculation for all VAEs"""
        for vae in self.adapter_vaes:
            vae.set_wasserstein_params(normalize, norm_method)

    def should_add_adapter(self, x, threshold=0.5, domain_name="default"):
        """Determine whether to add new adapter, based on minimum distance to historical domains"""
        if len(self.adapter_vaes) == 0:
            return True

        # Collect all historical domains
        historical_domains = set()
        for vae in self.adapter_vaes:
            historical_domains.update(vae.saved_distributions.keys())
        historical_domains.discard(domain_name)  # Exclude current domain

        if not historical_domains:
            return True  # No historical domains, add

        distances = []
        for domain in historical_domains:
            # For each historical domain, take minimum distance of all VAEs
            domain_distances = []
            for vae in self.adapter_vaes:
                if domain in vae.saved_distributions:
                    dist = vae.compute_distribution_distance(x, domain)
                    domain_distances.append(dist)
            if domain_distances:
                distances.append(min(domain_distances))

        if not distances:
            return True

        min_distance = min(distances)
        return min_distance > threshold

    def get_num_adapters(self):
        """Get current adapter count"""
        return len(self.adapters)

    def get_adapter_weights(self):
        """Get weights of all adapters (compatible with old interface)

        If no input features are passed, return L2 norm of each routing vector (for diagnosis).
        Actual routing is computed dynamically during forward pass via get_adapter_weights(x).
        """
        if len(self.adapter_weights) == 0:
            device = (
                next(self.parameters()).device
                if len(list(self.parameters())) > 0
                else "cpu"
            )
            return torch.tensor([], device=device)

        # Return L2 norm of each column vector (static, for diagnosis)
        vecs = torch.stack([w for w in self.adapter_weights], dim=1)  # (D, K)
        norms = torch.norm(vecs, dim=0)
        return norms.view(-1)

    def get_adapter_weights(self, x: torch.Tensor):
        """Dynamically compute weights for each adapter based on input x (literature-style routing)

        Args:
            x: tensor of shape [B, S, D] (batch, seq_len, dim)

        Returns:
            weights: tensor of shape [B, K] (per-sample softmax weights over K adapters)
        """
        K = len(self.adapter_weights)
        device = next(self.parameters()).device
        if K == 0:
            return torch.zeros((x.shape[0], 0), device=device)

        # Stack to W ∈ R^{D×K}
        W = torch.stack([w for w in self.adapter_weights], dim=1)  # (D, K)

        # Normalize input and routing vectors
        x_norm = F.normalize(x, dim=-1)  # [B, S, D]
        W_norm = F.normalize(W, dim=0)  # [D, K]

        # Cosine similarity
        logits = torch.matmul(x_norm, W_norm)  # (B, S, K)

        # Aggregate sequence dimension (take mean), get K-dimensional logits for each sample
        logits_mean = logits.mean(dim=1)  # (B, K)

        # Temperature scaling (temp < 1 makes distribution sharper)
        temp = 0.1  # Adjustable hyperparameter
        weights = F.softmax(logits_mean / temp, dim=-1)  # (B, K)
        return weights

    def get_vae(self, index):
        """Get VAE of specified index"""
        if index < len(self.adapter_vaes):
            return self.adapter_vaes[index]
        return None

    def get_adapter_input_features(self, index):
        """Get adapter input features of specified index"""
        if index < len(self.adapter_input_features):
            return self.adapter_input_features[index]
        return None

    def initialize_new_adapter_features(self, sample_dataloader):
        """Initialize feature storage for newly added adapter (using a small batch)"""
        if any(features is None for features in self.adapter_input_features):
            # Temporarily set to eval mode, perform one forward pass to initialize features
            was_training = self.training
            self.eval()

            with torch.no_grad():
                try:
                    # Get a small batch
                    sample_batch = next(iter(sample_dataloader))
                    inputs = sample_batch["video"]

                    # Ensure on correct device
                    device = next(self.parameters()).device
                    inputs = inputs.to(device)

                    # Here we need to simulate forward pass input to this layer
                    # Since we're in AdapterList, need to get features from external
                    print(
                        f"Warning: initialize_new_adapter_features needs to be called from model level"
                    )

                except Exception as e:
                    print(f"Error initializing adapter features: {e}")
                finally:
                    # Restore original training state
                    if was_training:
                        self.train()

    def forward(self, x):
        # Store input features for each adapter for VAE training
        # Store features in training mode, or when any adapter has no stored features
        if self.training or any(
            features is None for features in self.adapter_input_features
        ):
            for i in range(len(self.adapters)):
                if self.training or self.adapter_input_features[i] is None:
                    self.adapter_input_features[i] = x.detach().clone()

        if len(self.adapters) == 0:
            return torch.zeros_like(x)

        # Dynamic routing: calculate adapter weights for each sample based on input x
        # x shape: [B, S, D]
        if len(self.adapters) == 0:
            return torch.zeros_like(x)

        weights = self.get_adapter_weights(x)  # (B, K)
        if weights.numel() == 0:
            return torch.zeros_like(x)

        K = len(self.adapters)
        B, S, D = x.shape
        weighted_output = torch.zeros_like(x)
        for i, adapter in enumerate(self.adapters):
            out_i = adapter(x)  # (B, S, D)
            w_i = weights[:, i].view(B, 1, 1)  # (B,1,1)
            weighted_output = weighted_output + w_i * out_i

        return weighted_output


class ViT(nn.Module):
    def __init__(
        self,
        patches: int = 16,
        dim: int = 768,
        ff_dim: int = 3072,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout_rate: float = 0.2,
        in_channels: int = 3,
        num_frames: int = 160,
        theta: float = 0.2,
        image_size: Optional[int] = None,
        # New adapter related parameters
        use_adapters: bool = True,
        initial_adapters: int = 1,
        vae_latent_dim: int = 64,
        vae_hidden_dim: int = 256,
        adapter_threshold: float = 0.1,  # VAE distance threshold
        backbone_weight: float = 1.0,
        adapter_weight: float = 0.1,
    ):
        super().__init__()

        self.frame = num_frames
        self.dim = dim
        self.use_adapters = use_adapters
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dim = vae_hidden_dim
        self.adapter_threshold = adapter_threshold
        self.current_domain = "domain_0"  # Current domain name
        self.backbone_weight = backbone_weight
        self.adapter_weight = adapter_weight
        self.accumulated_fisher = (
            None  # Used to store accumulated Fisher information matrix
        )

        # Image and patch sizes
        ft, fh, fw = as_tuple(patches)  # patch sizes, ft = 4 ==> 160/4=40

        # feature_embedding Block
        self.feature_embedding = FeatureEmbedding(in_channels, dim)

        # Patch embedding    [4x16x16]conv
        self.patch_embedding = nn.Conv3d(
            dim, dim, kernel_size=(ft, fh, fw), stride=(ft, fh, fw)
        )

        # Transformer
        self.vit_block_head = ViTBackbone(
            num_layers=num_layers - 3,
            dim=dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate,
        )

        self.vit_block_last_1 = ViTBackbone(
            num_layers=1,
            dim=dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate,
        )

        self.vit_block_last_2 = ViTBackbone(
            num_layers=1,
            dim=dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate,
        )

        self.vit_block_last_3 = ViTBackbone(
            num_layers=1,
            dim=dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout_rate,
        )

        # Add AdapterList to last 3 layers
        if self.use_adapters:
            self.adapter_list_1 = AdapterList(
                dim,
                dropout=dropout_rate,
                initial_adapters=initial_adapters,
                vae_latent_dim=vae_latent_dim,
                vae_hidden_dim=vae_hidden_dim,
            )
            self.adapter_list_2 = AdapterList(
                dim,
                dropout=dropout_rate,
                initial_adapters=initial_adapters,
                vae_latent_dim=vae_latent_dim,
                vae_hidden_dim=vae_hidden_dim,
            )
            self.adapter_list_3 = AdapterList(
                dim,
                dropout=dropout_rate,
                initial_adapters=initial_adapters,
                vae_latent_dim=vae_latent_dim,
                vae_hidden_dim=vae_hidden_dim,
            )

        self.predictor = Predictor(dim)

    def save_domain_distributions(self, dataloader, domain_name=None):
        """Save current domain distribution information"""
        if not self.use_adapters:
            return

        if domain_name is None:
            domain_name = self.current_domain

        self.eval()
        with torch.no_grad():
            sample_batch = next(iter(dataloader))
            x = sample_batch["video"]

            # Ensure data is on correct device
            device = next(self.parameters()).device
            x = x.to(device)

            # Forward pass to get intermediate features
            x = torch.diff(x, n=1, dim=2, prepend=x[:, :, 0:1, :, :])
            stem_feat = self.feature_embedding(x)
            x = self.patch_embedding(stem_feat)
            x = x.flatten(2).transpose(1, 2)
            # Head segment output
            x_head = self.vit_block_head(x)

            # Block 1: features after attention (pre-FFN)
            blk1 = self.vit_block_last_1.blocks[0]
            attn_out_1 = blk1.attn(blk1.norm1(x_head))
            h1 = blk1.drop(blk1.proj(attn_out_1))
            x_attn_1 = x_head + h1
            self.adapter_list_1.save_vae_distributions(x_attn_1, domain_name)

            # Calculate complete Block1 output, as Block2 input
            h1_ffn = blk1.drop(blk1.pwff(blk1.norm2(x_attn_1)))
            x_blk1_out = x_attn_1 + h1_ffn

            # Block 2: features after attention (pre-FFN)
            blk2 = self.vit_block_last_2.blocks[0]
            attn_out_2 = blk2.attn(blk2.norm1(x_blk1_out))
            h2 = blk2.drop(blk2.proj(attn_out_2))
            x_attn_2 = x_blk1_out + h2
            self.adapter_list_2.save_vae_distributions(x_attn_2, domain_name)

            h2_ffn = blk2.drop(blk2.pwff(blk2.norm2(x_attn_2)))
            x_blk2_out = x_attn_2 + h2_ffn

            # Block 3: features after attention (pre-FFN)
            blk3 = self.vit_block_last_3.blocks[0]
            attn_out_3 = blk3.attn(blk3.norm1(x_blk2_out))
            h3 = blk3.drop(blk3.proj(attn_out_3))
            x_attn_3 = x_blk2_out + h3
            self.adapter_list_3.save_vae_distributions(x_attn_3, domain_name)

    def _get_adapter_layer_map(self):
        """Return mapping from layer names to AdapterList"""
        if not self.use_adapters:
            return {}
        return {
            "layer_1": getattr(self, "adapter_list_1", None),
            "layer_2": getattr(self, "adapter_list_2", None),
            "layer_3": getattr(self, "adapter_list_3", None),
        }

    def _clear_saved_distributions(self):
        """Clear cached distributions in all VAEs"""
        for adapter_list in self._get_adapter_layer_map().values():
            if adapter_list is None:
                continue
            for vae in adapter_list.adapter_vaes:
                vae.saved_distributions.clear()

    def persist_domain_fingerprints(self, domain_name, storage_dir):
        """Persist fingerprints for each layer of current domain"""
        if not self.use_adapters:
            return
        os.makedirs(storage_dir, exist_ok=True)
        layer_map = self._get_adapter_layer_map()
        layers_data = {}
        for layer_key, adapter_list in layer_map.items():
            if adapter_list is None:
                continue
            normalized_mu = None
            for vae in adapter_list.adapter_vaes:
                entry = vae.saved_distributions.get(domain_name)
                if entry is not None:
                    normalized_mu = np.asarray(entry["normalized_mu"], dtype=np.float32)
                    break
            if normalized_mu is not None:
                layers_data[layer_key] = normalized_mu

        if not layers_data:
            return

        file_path = os.path.join(storage_dir, f"{domain_name}.pth")
        torch.save({"domain": domain_name, "layers": layers_data}, file_path)

    def reload_persisted_distributions(self, storage_dir):
        """Load fingerprints of all historical domains from disk"""
        if not self.use_adapters or not os.path.isdir(storage_dir):
            return

        files = sorted(glob.glob(os.path.join(storage_dir, "*.pth")))
        if not files:
            return

        self._clear_saved_distributions()
        layer_map = self._get_adapter_layer_map()

        for file_path in files:
            try:
                payload = torch.load(file_path)
            except Exception:
                continue
            domain_name = payload.get("domain")
            layers_data = payload.get("layers", {})
            if domain_name is None:
                continue

            for layer_key, normalized_mu in layers_data.items():
                adapter_list = layer_map.get(layer_key)
                if adapter_list is None:
                    continue
                mu_array = np.asarray(normalized_mu, dtype=np.float32).copy()
                for vae in adapter_list.adapter_vaes:
                    vae.saved_distributions[domain_name] = {"normalized_mu": mu_array}

    def set_wasserstein_params(self, normalize=False, norm_method="l2"):
        """Set normalization parameters for Wasserstein distance calculation for all AdapterLists"""
        if self.use_adapters:
            self.adapter_list_1.set_wasserstein_params(normalize, norm_method)
            self.adapter_list_2.set_wasserstein_params(normalize, norm_method)
            self.adapter_list_3.set_wasserstein_params(normalize, norm_method)

    def intelligent_expand(self, sample_dataloader, new_domain_name=None):
        """Intelligent expansion: decide whether to add adapter based on VAE distribution distance"""
        if not self.use_adapters:
            raise ValueError("Adapters are not enabled for this model")

        if new_domain_name is None:
            new_domain_name = f"domain_{len(set([domain for vae in self.adapter_list_1.adapter_vaes for domain in vae.saved_distributions.keys()]))}"

        # First freeze all existing adapters
        self.adapter_list_1.freeze_adapters()
        self.adapter_list_2.freeze_adapters()
        self.adapter_list_3.freeze_adapters()

        # Keep backbone trainable, only freeze existing adapters to focus on incremental modules

        self.eval()
        expansion_info = {}
        # Used for progressive regularization during training: record parent domain and initial W distance for each layer
        if not hasattr(self, "parent_info"):
            self.parent_info = {}

        with torch.no_grad():
            # Take a small batch for testing
            sample_batch = next(iter(sample_dataloader))
            x = sample_batch["video"]

            # Ensure data is on correct device
            device = next(self.parameters()).device
            x = x.to(device)

            # Forward pass to get intermediate features
            x = torch.diff(x, n=1, dim=2, prepend=x[:, :, 0:1, :, :])
            stem_feat = self.feature_embedding(x)
            x = self.patch_embedding(stem_feat)
            x = x.flatten(2).transpose(1, 2)

            x_head = self.vit_block_head(x)

            # Judge layer by layer whether to add adapter (using pre-FFN features after attention)
            # Also determine the closest parent domain for each layer (by minimum W distance) and inherit parameters when needed
            # Layer 1
            blk1 = self.vit_block_last_1.blocks[0]
            attn_out_1 = blk1.attn(blk1.norm1(x_head))
            h1 = blk1.drop(blk1.proj(attn_out_1))
            x_attn_1 = x_head + h1
            # Select closest parent domain (based on all saved domains, take minimum W distance)
            parent_domain_1 = None
            parent_w_1 = None
            # Summarize domain names saved in all VAEs of this layer
            domains_layer1 = set()
            for vae in self.adapter_list_1.adapter_vaes:
                domains_layer1.update(list(vae.saved_distributions.keys()))
            if new_domain_name in domains_layer1:
                domains_layer1.discard(new_domain_name)
            # Calculate minimum W distance for each domain (traverse all VAEs in this layer)
            historical_distances_1 = []
            for d in domains_layer1:
                dists = []
                for vae in self.adapter_list_1.adapter_vaes:
                    dists.append(vae.compute_distribution_distance(x_attn_1, d))
                if len(dists) > 0:
                    dw = min(dists)
                    historical_distances_1.append(dw)
                    if parent_w_1 is None or dw < parent_w_1:
                        parent_w_1 = dw
                        parent_domain_1 = d
            distances_1 = self.adapter_list_1.compute_vae_distances(
                x_attn_1, self.current_domain
            )
            should_add_1 = self.adapter_list_1.should_add_adapter(
                x_attn_1, self.adapter_threshold, self.current_domain
            )
            expansion_info["layer_1_distances"] = historical_distances_1
            if should_add_1:
                idx_1 = self.adapter_list_1.add_adapter()
                # If parent domain is found, inherit parameters from parent adapter
                parent_idx_1 = None
                if (
                    parent_domain_1 is not None
                    and len(self.adapter_list_1.adapter_vaes) > 1
                ):
                    # Among all adapters in this layer, pick the one with minimum W distance to parent domain as parent adapter
                    best_dist = None
                    best_idx = None
                    for i, vae in enumerate(
                        self.adapter_list_1.adapter_vaes[:-1]
                    ):  # The last one is newly added
                        dw = vae.compute_distribution_distance(
                            x_attn_1, parent_domain_1
                        )
                        if best_dist is None or dw < best_dist:
                            best_dist = dw
                            best_idx = i
                    parent_idx_1 = best_idx
                # Inherit parameters (if there is parent adapter)
                if parent_idx_1 is not None:
                    try:
                        src_adapter = self.adapter_list_1.adapters[parent_idx_1]
                        dst_adapter = self.adapter_list_1.adapters[idx_1]
                        dst_adapter.load_state_dict(src_adapter.state_dict())
                        src_vae = self.adapter_list_1.adapter_vaes[parent_idx_1]
                        dst_vae = self.adapter_list_1.adapter_vaes[idx_1]
                        dst_vae.load_state_dict(src_vae.state_dict())
                    except Exception:
                        pass
                # Record parent domain info and initial W distance (for progressive regularization radius during training)
                self.parent_info["layer_1"] = {
                    "domain": parent_domain_1,
                    "parent_idx": parent_idx_1,
                    "init_w": float(parent_w_1) if parent_w_1 is not None else None,
                }
                self.adapter_list_1.unfreeze_latest_adapter()
                expansion_info["layer_1"] = f"Added adapter {idx_1}"
            else:
                expansion_info["layer_1"] = "No adapter added"

            # Layer 2
            h1_ffn = blk1.drop(blk1.pwff(blk1.norm2(x_attn_1)))
            x_blk1_out = x_attn_1 + h1_ffn
            blk2 = self.vit_block_last_2.blocks[0]
            attn_out_2 = blk2.attn(blk2.norm1(x_blk1_out))
            h2 = blk2.drop(blk2.proj(attn_out_2))
            x_attn_2 = x_blk1_out + h2
            parent_domain_2 = None
            parent_w_2 = None
            domains_layer2 = set()
            for vae in self.adapter_list_2.adapter_vaes:
                domains_layer2.update(list(vae.saved_distributions.keys()))
            if new_domain_name in domains_layer2:
                domains_layer2.discard(new_domain_name)
            # Calculate minimum W distance for each domain (traverse all VAEs in this layer)
            historical_distances_2 = []
            for d in domains_layer2:
                dists = []
                for vae in self.adapter_list_2.adapter_vaes:
                    dists.append(vae.compute_distribution_distance(x_attn_2, d))
                if len(dists) > 0:
                    dw = min(dists)
                    historical_distances_2.append(dw)
                    if parent_w_2 is None or dw < parent_w_2:
                        parent_w_2 = dw
                        parent_domain_2 = d
            distances_2 = self.adapter_list_2.compute_vae_distances(
                x_attn_2, self.current_domain
            )
            should_add_2 = self.adapter_list_2.should_add_adapter(
                x_attn_2, self.adapter_threshold, self.current_domain
            )
            expansion_info["layer_2_distances"] = historical_distances_2
            if should_add_2:
                idx_2 = self.adapter_list_2.add_adapter()
                parent_idx_2 = None
                if (
                    parent_domain_2 is not None
                    and len(self.adapter_list_2.adapter_vaes) > 1
                ):
                    best_dist = None
                    best_idx = None
                    for i, vae in enumerate(self.adapter_list_2.adapter_vaes[:-1]):
                        dw = vae.compute_distribution_distance(
                            x_attn_2, parent_domain_2
                        )
                        if best_dist is None or dw < best_dist:
                            best_dist = dw
                            best_idx = i
                    parent_idx_2 = best_idx
                if parent_idx_2 is not None:
                    try:
                        src_adapter = self.adapter_list_2.adapters[parent_idx_2]
                        dst_adapter = self.adapter_list_2.adapters[idx_2]
                        dst_adapter.load_state_dict(src_adapter.state_dict())
                        src_vae = self.adapter_list_2.adapter_vaes[parent_idx_2]
                        dst_vae = self.adapter_list_2.adapter_vaes[idx_2]
                        dst_vae.load_state_dict(src_vae.state_dict())
                    except Exception:
                        pass
                self.parent_info["layer_2"] = {
                    "domain": parent_domain_2,
                    "parent_idx": parent_idx_2,
                    "init_w": float(parent_w_2) if parent_w_2 is not None else None,
                }
                self.adapter_list_2.unfreeze_latest_adapter()
                expansion_info["layer_2"] = f"Added adapter {idx_2}"
            else:
                expansion_info["layer_2"] = "No adapter added"

            # Layer 3
            h2_ffn = blk2.drop(blk2.pwff(blk2.norm2(x_attn_2)))
            x_blk2_out = x_attn_2 + h2_ffn
            blk3 = self.vit_block_last_3.blocks[0]
            attn_out_3 = blk3.attn(blk3.norm1(x_blk2_out))
            h3 = blk3.drop(blk3.proj(attn_out_3))
            x_attn_3 = x_blk2_out + h3
            parent_domain_3 = None
            parent_w_3 = None
            domains_layer3 = set()
            for vae in self.adapter_list_3.adapter_vaes:
                domains_layer3.update(list(vae.saved_distributions.keys()))
            if new_domain_name in domains_layer3:
                domains_layer3.discard(new_domain_name)
            # Calculate minimum W distance for each domain (traverse all VAEs in this layer)
            historical_distances_3 = []
            for d in domains_layer3:
                dists = []
                for vae in self.adapter_list_3.adapter_vaes:
                    dists.append(vae.compute_distribution_distance(x_attn_3, d))
                if len(dists) > 0:
                    dw = min(dists)
                    historical_distances_3.append(dw)
                    if parent_w_3 is None or dw < parent_w_3:
                        parent_w_3 = dw
                        parent_domain_3 = d
            distances_3 = self.adapter_list_3.compute_vae_distances(
                x_attn_3, self.current_domain
            )
            should_add_3 = self.adapter_list_3.should_add_adapter(
                x_attn_3, self.adapter_threshold, self.current_domain
            )
            expansion_info["layer_3_distances"] = historical_distances_3
            if should_add_3:
                idx_3 = self.adapter_list_3.add_adapter()
                parent_idx_3 = None
                if (
                    parent_domain_3 is not None
                    and len(self.adapter_list_3.adapter_vaes) > 1
                ):
                    best_dist = None
                    best_idx = None
                    for i, vae in enumerate(self.adapter_list_3.adapter_vaes[:-1]):
                        dw = vae.compute_distribution_distance(
                            x_attn_3, parent_domain_3
                        )
                        if best_dist is None or dw < best_dist:
                            best_dist = dw
                            best_idx = i
                    parent_idx_3 = best_idx
                if parent_idx_3 is not None:
                    try:
                        src_adapter = self.adapter_list_3.adapters[parent_idx_3]
                        dst_adapter = self.adapter_list_3.adapters[idx_3]
                        dst_adapter.load_state_dict(src_adapter.state_dict())
                        src_vae = self.adapter_list_3.adapter_vaes[parent_idx_3]
                        dst_vae = self.adapter_list_3.adapter_vaes[idx_3]
                        dst_vae.load_state_dict(src_vae.state_dict())
                    except Exception:
                        pass
                self.parent_info["layer_3"] = {
                    "domain": parent_domain_3,
                    "parent_idx": parent_idx_3,
                    "init_w": float(parent_w_3) if parent_w_3 is not None else None,
                }
                self.adapter_list_3.unfreeze_latest_adapter()
                expansion_info["layer_3"] = f"Added adapter {idx_3}"
            else:
                expansion_info["layer_3"] = "No adapter added"

        # Update current domain name
        self.current_domain = new_domain_name

        return expansion_info

    def expend(self):
        """Original expansion method, add adapter to all AdapterLists (maintain backward compatibility)"""
        if not self.use_adapters:
            raise ValueError("Adapters are not enabled for this model")

        new_adapter_indices = []
        idx_1 = self.adapter_list_1.add_adapter()
        idx_2 = self.adapter_list_2.add_adapter()
        idx_3 = self.adapter_list_3.add_adapter()

        new_adapter_indices.append(idx_1)
        new_adapter_indices.append(idx_2)
        new_adapter_indices.append(idx_3)

        return new_adapter_indices

    def get_adapter_info(self):
        """Get information of all AdapterLists"""
        if not self.use_adapters:
            return "Adapters are not enabled for this model"

        info = {
            "layer_1": self.adapter_list_1.get_num_adapters(),
            "layer_2": self.adapter_list_2.get_num_adapters(),
            "layer_3": self.adapter_list_3.get_num_adapters(),
            "total_adapters": (
                self.adapter_list_1.get_num_adapters()
                + self.adapter_list_2.get_num_adapters()
                + self.adapter_list_3.get_num_adapters()
            ),
            "current_domain": self.current_domain,
        }
        return info

    def initialize_new_adapters(self, sample_dataloader):
        """Initialize feature storage for newly added adapters"""
        if not self.use_adapters:
            return

        # Check if any adapter needs initialization
        need_init = False
        if hasattr(self.adapter_list_1, "adapter_input_features"):
            need_init = need_init or any(
                features is None
                for features in self.adapter_list_1.adapter_input_features
            )
        if hasattr(self.adapter_list_2, "adapter_input_features"):
            need_init = need_init or any(
                features is None
                for features in self.adapter_list_2.adapter_input_features
            )
        if hasattr(self.adapter_list_3, "adapter_input_features"):
            need_init = need_init or any(
                features is None
                for features in self.adapter_list_3.adapter_input_features
            )

        if not need_init:
            return

        print("Initializing features for newly added adapters...")

        # Temporarily set to eval mode, perform one forward pass to initialize features
        was_training = self.training
        self.eval()

        with torch.no_grad():
            try:
                # Get a small batch
                sample_batch = next(iter(sample_dataloader))
                inputs = sample_batch["video"]

                # Ensure on correct device
                device = next(self.parameters()).device
                inputs = inputs.to(device)

                # Execute forward pass to initialize features
                self.forward({"input_clip": inputs})

                print("Successfully initialized adapter features")

            except Exception as e:
                print(f"Error initializing adapter features: {e}")
            finally:
                # Restore original training state
                if was_training:
                    self.train()

    def forward(self, inputs):
        x = inputs["input_clip"]

        x = torch.diff(x, n=1, dim=2, prepend=x[:, :, 0:1, :, :])

        b, c, t, fh, fw = x.shape

        stem_feat = self.feature_embedding(x)

        x = self.patch_embedding(stem_feat)
        x = x.flatten(2).transpose(1, 2)

        # Front Transformer
        x = self.vit_block_head(x)

        # Block 1: Adapter after attention residual, then into FFN
        blk1 = self.vit_block_last_1.blocks[0]
        attn_out_1 = blk1.attn(blk1.norm1(x))
        h1 = blk1.drop(blk1.proj(attn_out_1))
        x_attn_1_pre = x + h1
        if self.use_adapters:
            adapter_out_1 = self.adapter_list_1(x_attn_1_pre)
            x_attn_1 = (
                self.backbone_weight * x_attn_1_pre
                + self.adapter_weight * adapter_out_1
            )
        else:
            x_attn_1 = x_attn_1_pre
        h1_ffn = blk1.drop(blk1.pwff(blk1.norm2(x_attn_1)))
        x = x_attn_1 + h1_ffn

        # Block 2
        blk2 = self.vit_block_last_2.blocks[0]
        attn_out_2 = blk2.attn(blk2.norm1(x))
        h2 = blk2.drop(blk2.proj(attn_out_2))
        x_attn_2_pre = x + h2
        if self.use_adapters:
            adapter_out_2 = self.adapter_list_2(x_attn_2_pre)
            x_attn_2 = (
                self.backbone_weight * x_attn_2_pre
                + self.adapter_weight * adapter_out_2
            )
        else:
            x_attn_2 = x_attn_2_pre
        h2_ffn = blk2.drop(blk2.pwff(blk2.norm2(x_attn_2)))
        x = x_attn_2 + h2_ffn

        # Block 3
        blk3 = self.vit_block_last_3.blocks[0]
        attn_out_3 = blk3.attn(blk3.norm1(x))
        h3 = blk3.drop(blk3.proj(attn_out_3))
        x_attn_3_pre = x + h3
        if self.use_adapters:
            adapter_out_3 = self.adapter_list_3(x_attn_3_pre)
            x_attn_3 = (
                self.backbone_weight * x_attn_3_pre
                + self.adapter_weight * adapter_out_3
            )
        else:
            x_attn_3 = x_attn_3_pre
        h3_ffn = blk3.drop(blk3.pwff(blk3.norm2(x_attn_3)))
        x = x_attn_3 + h3_ffn

        # upsampling heads
        features_last = x.transpose(1, 2).view(b, self.dim, t // 4, 4, 4)

        rPPG = self.predictor(features_last)

        outputs = {
            "rPPG": rPPG,
            "features_last": features_last,
            # Pre-FFN features used for progressive regularization during training
            "x_attn_1": x_attn_1_pre,
            "x_attn_2": x_attn_2_pre,
            "x_attn_3": x_attn_3_pre,
        }

        return outputs

    def unfreeze_backbone(self):
        """Unfreeze backbone model parameters"""
        # Unfreeze feature extraction layer
        for param in self.feature_embedding.parameters():
            param.requires_grad = True

        # Unfreeze patch embedding
        for param in self.patch_embedding.parameters():
            param.requires_grad = True

        # Unfreeze transformer backbone
        for param in self.vit_block_head.parameters():
            param.requires_grad = True
        for param in self.vit_block_last_1.parameters():
            param.requires_grad = True
        for param in self.vit_block_last_2.parameters():
            param.requires_grad = True
        for param in self.vit_block_last_3.parameters():
            param.requires_grad = True

        # Unfreeze prediction head
        for param in self.predictor.parameters():
            param.requires_grad = True

    def random_freeze_backbone(self, freeze_ratio=0.0):
        """Randomly freeze a portion of transformer backbone parameters

        Args:
            freeze_ratio (float): Freeze ratio, 0.0 means no freeze, 1.0 means freeze all
        """
        if freeze_ratio <= 0.0:
            return

        # Get all parameters of transformer backbone
        backbone_params = []
        for module in [
            self.vit_block_head,
            self.vit_block_last_1,
            self.vit_block_last_2,
            self.vit_block_last_3,
        ]:
            for param in module.parameters():
                backbone_params.append(param)

        # Randomly select parameters to freeze
        num_params = len(backbone_params)
        num_to_freeze = int(num_params * freeze_ratio)

        if num_to_freeze > 0:
            indices_to_freeze = random.sample(range(num_params), num_to_freeze)
            for idx in indices_to_freeze:
                backbone_params[idx].requires_grad = False

    def compute_and_store_fisher(
        self, dataloader, domain_name, Lc_weight=1.0, new_weight=0.5, old_weight=0.5
    ):
        """Calculate Fisher information matrix for current domain and weighted average with previous accumulated Fisher

        Args:
            dataloader: Data loader for calculating Fisher
            domain_name: Current domain name
            Lc_weight: Lc loss weight
            new_weight: New Fisher weight
            old_weight: Old Fisher weight
        """
        self.eval()
        fisher_info = {}

        # Get backbone parameters
        backbone_params = []
        param_names = []
        for name, module in [
            ("vit_block_head", self.vit_block_head),
            ("vit_block_last_1", self.vit_block_last_1),
            ("vit_block_last_2", self.vit_block_last_2),
            ("vit_block_last_3", self.vit_block_last_3),
        ]:
            for param_name, param in module.named_parameters():
                if param.requires_grad:  # Only calculate trainable parameters
                    backbone_params.append(param)
                    param_names.append(f"{name}.{param_name}")

        # Initialize Fisher
        for param in backbone_params:
            fisher_info[param] = torch.zeros_like(param)

        # Calculate Fisher: use augmentation consistency loss Lc
        for batch in dataloader:
            inputs = batch["video"]
            device = next(self.parameters()).device
            inputs = inputs.to(device)

            # Two forward passes to simulate augmentation consistency, using different random augmentations
            inputs_np = inputs.cpu().numpy()  # [B, C, T, H, W]

            def apply_random_augmentation(clip):
                # clip: [C, T, H, W] numpy, [0,1]
                clip = clip * 255.0  # Convert to [0,255]
                # Randomly apply augmentations
                if np.random.rand() < 0.5:
                    clip = transforms.augment_horizontal_flip(clip)
                if np.random.rand() < 0.5:
                    clip = transforms.augment_illumination_noise(clip)
                if np.random.rand() < 0.5:
                    clip = transforms.augment_gaussian_noise(clip)
                # Normalize to [0,1]
                clip = np.clip(clip, 0, 255) / 255.0
                return clip

            inputs1_np = np.zeros_like(inputs_np)
            inputs2_np = np.zeros_like(inputs_np)
            for b in range(inputs_np.shape[0]):
                clip = inputs_np[b]  # [C, T, H, W]
                inputs1_np[b] = apply_random_augmentation(clip.copy())
                inputs2_np[b] = apply_random_augmentation(clip.copy())

            inputs1 = torch.from_numpy(inputs1_np).float().to(device)
            inputs2 = torch.from_numpy(inputs2_np).float().to(device)

            outputs1 = self.forward({"input_clip": inputs1})
            outputs2 = self.forward({"input_clip": inputs2})

            # Lc: MSE between two predictions
            Lc = F.mse_loss(outputs1["rPPG"], outputs2["rPPG"])
            Lc *= Lc_weight

            # Calculate gradients
            self.zero_grad()
            Lc.backward(retain_graph=True)

            # Accumulate Fisher (gradient squared)
            for param in backbone_params:
                if param.grad is not None:
                    fisher_info[param] += param.grad.data**2

            break  # Use only one batch to simplify calculation

        # Normalize Fisher
        num_samples = 1  # Simplified
        for param in backbone_params:
            fisher_info[param] /= num_samples

        # Weighted average with previous accumulated Fisher
        if self.accumulated_fisher is None:
            self.accumulated_fisher = fisher_info
        else:
            for param in backbone_params:
                if param in self.accumulated_fisher:
                    self.accumulated_fisher[param] = (
                        new_weight * fisher_info[param]
                        + old_weight * self.accumulated_fisher[param]
                    )
                else:
                    self.accumulated_fisher[param] = fisher_info[param]

    def adaptive_freeze_backbone_from_stored(self, freeze_ratio=0.5):
        """Adaptively freeze backbone parameters based on accumulated Fisher

        Args:
            freeze_ratio (float): Freeze ratio
        """
        if self.accumulated_fisher is None:
            print("No accumulated Fisher available, falling back to random freeze")
            self.random_freeze_backbone(freeze_ratio)
            return

        # Get backbone parameters and corresponding Fisher values
        backbone_params = []
        fisher_values = []
        for name, module in [
            ("vit_block_head", self.vit_block_head),
            ("vit_block_last_1", self.vit_block_last_1),
            ("vit_block_last_2", self.vit_block_last_2),
            ("vit_block_last_3", self.vit_block_last_3),
        ]:
            for param_name, param in module.named_parameters():
                if param.requires_grad and param in self.accumulated_fisher:
                    backbone_params.append(param)
                    # Use Fisher mean as importance measure
                    fisher_values.append(self.accumulated_fisher[param].mean().item())

        if len(backbone_params) == 0:
            return

        # Freeze the most important parameters first, as measured by Fisher.
        sorted_indices = sorted(
            range(len(fisher_values)), key=lambda i: fisher_values[i], reverse=True
        )

        # Select parameters to freeze
        num_to_freeze = int(len(backbone_params) * freeze_ratio)
        if num_to_freeze > 0:
            for i in range(num_to_freeze):
                idx = sorted_indices[i]
                backbone_params[idx].requires_grad = False
