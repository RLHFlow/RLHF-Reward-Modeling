# %% Type definitions and base classes
from typing import Optional, Sequence, Dict, Any, Tuple, TypeVar, Protocol, Generic, Union, Callable
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing_extensions import Protocol
import abc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Array = torch.Tensor
PRNGKey = np.ndarray
Index = Any
Params = Dict[str, torch.Tensor]

State = Dict[str, Any]
LossMetrics = Dict[str, torch.Tensor]

@dataclass
class OutputWithPrior:
    train: torch.Tensor
    prior: torch.Tensor = torch.zeros(1)
    extra: Dict[str, torch.Tensor] = None

    @property
    def preds(self) -> torch.Tensor:
        return self.train + self.prior.detach()

NetworkOutput = Union[torch.Tensor, OutputWithPrior]

# %% Core MLP components
class ExposedMLP(nn.Module):
    """MLP that exposes internal layer features."""
    def __init__(self,
                 output_sizes: Sequence[int],
                 expose_layers: Optional[Sequence[bool]] = None,
                 stop_gradient: bool = True,
                 name: Optional[str] = None):
        super().__init__()
        self.name = name or "ExposedMLP"

        self.layers = nn.ModuleList()
        input_size = output_sizes[0]  # First element is input size
        for output_size in output_sizes[1:]:
            self.layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

        self.num_layers = len(self.layers)
        self.output_size = output_sizes[-1]
        self.stop_gradient = stop_gradient
        self.expose_layers = expose_layers or [True] * len(self.layers)
        assert len(self.expose_layers) == len(self.layers)

    def forward(self, inputs: torch.Tensor) -> OutputWithPrior:
        logger.info(f"{self.name} forward pass - Input shape: {inputs.shape}")

        layers_features = []
        out = inputs

        for i, layer in enumerate(self.layers):
            out = layer(out)
            logger.info(f"{self.name} layer {i} output shape: {out.shape}")
            if i < (self.num_layers - 1):
                out = torch.relu(out)
            layers_features.append(out)

        exposed_features = [inputs]
        for i, layer_feature in enumerate(layers_features):
            if self.expose_layers[i]:
                exposed_features.append(layer_feature)

        exposed_features = torch.cat(exposed_features, dim=1)
        if self.stop_gradient:
            exposed_features = exposed_features.detach()

        logger.info(f"{self.name} final output shape: {out.shape}")
        logger.info(f"{self.name} exposed features shape: {exposed_features.shape}")

        return OutputWithPrior(
            train=out,
            prior=torch.zeros_like(out),
            extra={'exposed_features': exposed_features}
        )

class ProjectedMLP(nn.Module):
    """MLP with final layer projected using index."""
    def __init__(self,
                 hidden_sizes: Sequence[int],
                 final_out: int,
                 index_dim: int,
                 name: Optional[str] = None):
        super().__init__()
        self.name = name or "ProjectedMLP"

        layers = []
        input_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU()
            ])
            input_size = hidden_size

        layers.append(nn.Linear(input_size, final_out * index_dim))
        self.mlp = nn.Sequential(*layers)

        self.final_out = final_out
        self.index_dim = index_dim

    def forward(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        logger.info(f"{self.name} forward pass - Input shape: {inputs.shape}, Index shape: {index.shape}")

        assert index.shape == (self.index_dim,)

        output = self.mlp(inputs)
        logger.info(f"{self.name} MLP output shape: {output.shape}")

        reshaped_output = output.view(inputs.shape[0], self.final_out, self.index_dim)
        logger.info(f"{self.name} reshaped output shape: {reshaped_output.shape}")

        final_output = torch.matmul(reshaped_output, index)
        logger.info(f"{self.name} final output shape: {final_output.shape}")

        return final_output

class GaussianIndexer:
    """Generates Gaussian random indices."""
    def __init__(self, index_dim: int):
        self.index_dim = index_dim

    def __call__(self, key: Any) -> torch.Tensor:
        logger.info(f"GaussianIndexer generating index with dimension {self.index_dim}")
        return torch.randn(self.index_dim)

# %% Main Epinet architecture
class MLPEpinet(nn.Module):
    """Complete MLP epinet combining base network and epistemic components."""
    def __init__(self,
                 output_sizes: Sequence[int],
                 epinet_hiddens: Sequence[int],
                 index_dim: int,
                 expose_layers: Optional[Sequence[bool]] = None,
                 prior_scale: float = 1.,
                 stop_gradient: bool = False,
                 name: Optional[str] = None):
        super().__init__()
        self.name = name or "MLPEpinet"

        prefix = f"{name}_" if name else ""

        self.base_mlp = ExposedMLP(
            output_sizes,
            expose_layers,
            stop_gradient,
            name=prefix+'base_mlp'
        )

        num_classes = output_sizes[-1]
        self.train_epinet = ProjectedMLP(
            epinet_hiddens,
            num_classes,
            index_dim,
            name=prefix+'train_epinet'
        )
        self.prior_epinet = ProjectedMLP(
            epinet_hiddens,
            num_classes,
            index_dim,
            name=prefix+'prior_epinet'
        )

        self.prior_scale = prior_scale
        self.stop_gradient = stop_gradient

    def forward(self, x: torch.Tensor, z: Index) -> OutputWithPrior:
        logger.info(f"{self.name} forward pass - Input shape: {x.shape}, Index shape: {z.shape}")

        base_out = self.base_mlp(x)
        features = base_out.extra['exposed_features']
        logger.info(f"{self.name} base network features shape: {features.shape}")

        if self.stop_gradient:
            epi_inputs = features.detach()
        else:
            epi_inputs = features

        epi_train = self.train_epinet(epi_inputs, z)
        epi_prior = self.prior_epinet(epi_inputs, z)

        logger.info(f"{self.name} final output shapes - Train: {epi_train.shape}, Prior: {epi_prior.shape}")

        return OutputWithPrior(
            train=base_out.train + epi_train,
            prior=self.prior_scale * epi_prior,
        )

def make_mlp_epinet(
    output_sizes: Sequence[int],
    epinet_hiddens: Sequence[int],
    index_dim: int,
    expose_layers: Optional[Sequence[bool]] = None,
    prior_scale: float = 1.,
    stop_gradient: bool = False,
    name: Optional[str] = None,
) -> Tuple[nn.Module, Any]:
    """Factory function to create a standard MLP epinet."""
    logger.info(f"Creating MLPEpinet with output sizes: {output_sizes}, epinet hiddens: {epinet_hiddens}, index dim: {index_dim}")

    model = MLPEpinet(
        output_sizes=output_sizes,
        epinet_hiddens=epinet_hiddens,
        index_dim=index_dim,
        expose_layers=expose_layers,
        prior_scale=prior_scale,
        stop_gradient=stop_gradient,
        name=name
    )

    indexer = GaussianIndexer(index_dim)

    return model, indexer


# %% Usage example
def run_epinet_example():
    """Example usage of MLP epinet."""
    logger.info("Running epinet example")

    # Create model
    input_dim = 10
    output_dim = 2
    model, indexer = make_mlp_epinet(
        output_sizes=[input_dim, 64, 32, output_dim],
        epinet_hiddens=[128, 64],
        index_dim=8
    )

    # Generate sample data
    batch_size = 4
    x = torch.randn(batch_size, input_dim)
    z = indexer(None)

    # Forward pass
    output = model(x, z)

    # Access outputs
    train_pred = output.train  # Shape: [batch_size, output_dim]
    prior_pred = output.prior  # Shape: [batch_size, output_dim]


# %% example testing
# Create model
input_dim = 10
output_dim = 2
output_sizes = [input_dim, 64, 32, output_dim]
index_dim = 8

model, indexer = make_mlp_epinet(
    output_sizes=output_sizes,
    epinet_hiddens=[input_dim, index_dim],
    index_dim=index_dim,
    expose_layers=[False] + [False]*(len(output_sizes)-2 -1) + [False]
)

# Generate sample data
batch_size = 4
x = torch.randn(batch_size, input_dim)
z = indexer(None)

print(z.shape)

# Forward pass
output = model(x, z)

# Access outputs
train_pred = output.train  # Shape: [batch_size, output_dim]
prior_pred = output.prior  # Shape: [batch_size, output_dim]
