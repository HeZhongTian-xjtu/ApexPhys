import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(object):
    def __init__(self, kl_weight=0.001, print_info=True):
        """
        VAE loss function

        Args:
            kl_weight (float): Weight of KL divergence loss, default is 0.001
            print_info (bool): Whether to print detailed information
        """
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        self.print_info = print_info

    def compute_vae_loss(self, features, vae):
        """
        Compute reconstruction loss for a single VAE

        Args:
            features (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            vae (nn.Module): VAE model

        Returns:
            tuple: (total_loss, recon_loss, kl_loss)
        """
        # Reshape features to VAE input format [batch_size * seq_len, feature_dim]
        batch_size, seq_len, feature_dim = features.shape
        features_flat = features.reshape(-1, feature_dim)

        # VAE forward pass
        recon_x, mu, log_var = vae(features_flat)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, features_flat, reduction="mean")

        # Calculate KL divergence loss
        kl_loss = (
            -0.5
            * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            / features_flat.size(0)
        )

        # Total VAE loss
        total_loss = recon_loss + self.kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def __call__(self, model):
        """
        Compute VAE loss for all AdapterLists in the model

        Args:
            model (nn.Module): Model containing AdapterList

        Returns:
            torch.Tensor: Total VAE loss
        """
        total_vae_loss = 0.0
        num_adapters = 0

        # Used to store VAE training information
        vae_info = {}

        # Get all AdapterList in model
        adapter_lists = []
        if hasattr(model, "adapter_list_1") and model.adapter_list_1 is not None:
            adapter_lists.append(("layer_1", model.adapter_list_1))
        if hasattr(model, "adapter_list_2") and model.adapter_list_2 is not None:
            adapter_lists.append(("layer_2", model.adapter_list_2))
        if hasattr(model, "adapter_list_3") and model.adapter_list_3 is not None:
            adapter_lists.append(("layer_3", model.adapter_list_3))

        for layer_name, adapter_list in adapter_lists:
            if hasattr(adapter_list, "adapter_vaes") and hasattr(
                adapter_list, "adapter_input_features"
            ):
                layer_vaes = []

                # Calculate VAE reconstruction loss for each adapter, using respective input features
                for i, vae in enumerate(adapter_list.adapter_vaes):
                    adapter_features = adapter_list.get_adapter_input_features(i)

                    if adapter_features is not None:
                        total_loss, recon_loss, kl_loss = self.compute_vae_loss(
                            adapter_features, vae
                        )
                        total_vae_loss += total_loss
                        num_adapters += 1

                        # Collect VAE information
                        layer_vaes.append(
                            {
                                "adapter_idx": i,
                                "recon_loss": recon_loss.item(),
                                "kl_loss": kl_loss.item(),
                                "total_loss": total_loss.item(),
                            }
                        )
                    else:
                        # If no input features, record as untrained state
                        layer_vaes.append(
                            {
                                "adapter_idx": i,
                                "recon_loss": 0.0,
                                "kl_loss": 0.0,
                                "total_loss": 0.0,
                                "status": "no_features",
                            }
                        )

                if layer_vaes:
                    vae_info[layer_name] = layer_vaes

        # Print VAE training information (only during training and when print_info is enabled)
        if self.print_info and model.training and vae_info:
            print(f"VAE Training Status:")
            for layer_name, layer_vaes in vae_info.items():
                # Only calculate average for VAEs with features
                valid_vaes = [v for v in layer_vaes if "status" not in v]
                if valid_vaes:
                    avg_recon = sum(v["recon_loss"] for v in valid_vaes) / len(
                        valid_vaes
                    )
                    avg_kl = sum(v["kl_loss"] for v in valid_vaes) / len(valid_vaes)
                    avg_total = sum(v["total_loss"] for v in valid_vaes) / len(
                        valid_vaes
                    )
                    print(
                        f"  {layer_name}: {len(valid_vaes)}/{len(layer_vaes)} VAEs - Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}, Avg Total: {avg_total:.4f}"
                    )
                else:
                    print(
                        f"  {layer_name}: 0/{len(layer_vaes)} VAEs - No valid features"
                    )

        # Return average VAE loss
        if num_adapters > 0:
            return total_vae_loss / num_adapters
        else:
            # If no adapters, return zero loss
            return torch.tensor(0.0, requires_grad=True)
