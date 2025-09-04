import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121


# DENSENET BACKBONE


class OptimizedDenseNet121(nn.Module):
    """
    DenseNet121 backbone optimized for CIFAR-10 (32x32) input.
    - Modifies the initial convolution to handle smaller image sizes.
    - Replaces the final classifier with a linear layer to output a feature vector.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        # Load pretrained DenseNet121
        self.densenet = densenet121(weights='IMAGENET1K_V1')
        
        # Modify first conv layer for 32x32 input (instead of 224x224)
        self.densenet.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Replace classifier to output desired feature dimension
        self.densenet.classifier = nn.Linear(1024, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features."""
        return self.densenet(x)


# TOP-K MOE COMPONENTS


class SimpleExpert(nn.Module):
    """
    A simple MLP expert for the Top-K MoE baseline.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input features through the expert MLP."""
        return self.layers(x)

class TopKRouter(nn.Module):
    """
    Standard Top-K router that selects the top 'k' experts for each input.
    Includes a load balancing loss to encourage uniform expert utilization.
    """
    def __init__(self, input_dim: int, num_experts: int, k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes routing weights and load balancing loss.
        """
        gate_logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        routing_weights = torch.zeros_like(gate_logits)
        routing_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        load_balance_loss = self._calculate_load_balance_loss(routing_weights)
        
        return routing_weights, load_balance_loss
    
    def _calculate_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Calculate L2 load balancing loss."""
        if not self.training:
            return torch.tensor(0.0, device=routing_weights.device)
        
        expert_usage = routing_weights.sum(dim=0) / routing_weights.sum()
        target_usage = 1.0 / self.num_experts
        load_balance_loss = torch.sum((expert_usage - target_usage) ** 2)
        
        return load_balance_loss


# TOP-K MOE MODEL

class TopKMoEModel(nn.Module):
    """
    A simplified Top-K Mixture of Experts model for CIFAR-10 without a residual connection.
    """
    def __init__(self, feature_dim: int, num_classes: int, num_experts: int, k: int):
        super().__init__()
        self.backbone = OptimizedDenseNet121(feature_dim)
        self.router = TopKRouter(feature_dim, num_experts, k)
        self.experts = nn.ModuleList([
            SimpleExpert(feature_dim, num_classes) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MoE model.
        """
        features = self.backbone(x)
        routing_weights, load_balance_loss = self.router(features)
        expert_outputs = torch.stack([expert(features) for expert in self.experts], dim=1)
        
        # Weighted combination of expert outputs
        final_output = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=1)
        
        return final_output, routing_weights, features, load_balance_loss