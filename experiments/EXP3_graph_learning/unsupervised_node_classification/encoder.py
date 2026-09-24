"""Two-tower encoder and feature augmentation used by the fixed EXP3 protocol."""
from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform

class GConv(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GCNConv(input_dim if index == 0 else hidden_dim, hidden_dim)
                for index in range(num_layers)
            ]
        )
        self.activation = nn.PReLU(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x, edge_index))
        return x

def drop_feature(x: torch.Tensor, probability: float) -> torch.Tensor:
    mask = torch.rand(x.size(1), device=x.device) < probability
    augmented = x.clone()
    augmented[:, mask] = 0
    return augmented

class OriginalMomentEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_probability: float):
        super().__init__()
        self.encoder1 = GConv(input_dim, hidden_dim)
        self.encoder2 = GConv(input_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)
        self.pf = feature_probability

    @staticmethod
    def corruption(x: torch.Tensor) -> torch.Tensor:
        return x[torch.randperm(x.size(0), device=x.device)]

    def forward(
        self,
        x: torch.Tensor,
        original_edges: torch.Tensor,
        augmented_edges: torch.Tensor,
    ):
        x1, x2 = drop_feature(x, self.pf), drop_feature(x, self.pf)
        z1 = self.encoder1(x1, original_edges)
        z2 = self.encoder2(x2, augmented_edges)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(self.corruption(x1), original_edges)
        z2n = self.encoder2(self.corruption(x2), augmented_edges)
        return z1, z2, g1, g2, z1n, z2n

    def encode_clean(
        self, x: torch.Tensor, original_edges: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode unmasked features on the original graph with both towers."""
        return (
            self.encoder1(x, original_edges),
            self.encoder2(x, original_edges),
        )
