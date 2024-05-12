from typing import Dict, Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from mace.modules.models import ScaleShiftMACE
from mace.modules.utils import get_edge_vectors_and_lengths


@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self, irreps_in: o3.Irreps, hidden_irreps: o3.Irreps, irreps_out: o3.Irreps
    ):
        super().__init__()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=hidden_irreps)
        self.activation = torch.nn.SiLU()
        self.linear_2 = o3.Linear(irreps_in=hidden_irreps, irreps_out=irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear_2(self.activation(self.linear_1(x)))  # [n_nodes, irreps_out]


@compile_mode("script")
class HydroMACE(ScaleShiftMACE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_irreps: o3.Irreps = self.readouts[0].linear.irreps_in
        readouts = []
        for i in range(self.num_interactions):
            if i == 0:
                irreps_in = hidden_irreps
            else:
                irreps_in = (
                    str(hidden_irreps[0])
                    if i == self.num_interactions - 1
                    else hidden_irreps
                )
            readouts.append(
                NonLinearReadoutBlock(
                    irreps_in=irreps_in,
                    hidden_irreps=kwargs["MLP_irreps"],
                    irreps_out=o3.Irreps("5x0e"),
                )
            )
        self.readouts = torch.nn.ModuleList(readouts)

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interactions
        layer_contributions = []
        for interaction, product, readout in zip(
            self.interactions, self.products, self.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_energies = readout(node_feats)  # [n_nodes, 5]
            layer_contributions.append(node_energies)

        # Sum over energy contributions
        layer_contributions = torch.stack(
            layer_contributions, dim=-1
        )  # [n_nodes, 5, num_interactions]
        missing_hydrogen_logits = torch.sum(layer_contributions, dim=-1)  # [n_nodes, 5]
        missing_hydrogen_probs = torch.nn.functional.softmax(
            missing_hydrogen_logits, dim=-1
        )  # [n_nodes, 5]
        missing_hydrogens = torch.argmax(missing_hydrogen_probs, dim=-1)  # [n_nodes]
        return {
            "missing_hydrogen_logits": missing_hydrogen_logits,  # [n_nodes, 5]
            "missing_hydrogens": missing_hydrogens,
        }
