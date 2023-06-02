from typing import Dict, Optional

import torch
from e3nn.util.jit import compile_mode
from mace.modules.models import ScaleShiftMACE


@compile_mode("script")
class HydroMACE(ScaleShiftMACE):
    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        data["node_attrs"].requires_grad_(True)
        data["positions"].requires_grad_(True)

        # Embeddings
        node_feats, edge_attrs, edge_feats = self._get_initial_embeddings(data)

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
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            layer_contributions.append(node_energies)

        # Sum over energy contributions
        layer_contributions = torch.stack(layer_contributions, dim=-1)
        missing_hydrogens = torch.sum(layer_contributions, dim=-1)  # [n_nodes, ]

        return {
            "missing_hydrogens": missing_hydrogens,
        }
