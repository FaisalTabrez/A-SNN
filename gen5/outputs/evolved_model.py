"""
AMMC-SNN PyTorch/snnTorch export.

Google Colab setup:
    1. Run this in a notebook cell: !pip -q install snntorch
    2. Paste this script into the next cell, or upload evolved_model.py and run:
       %run evolved_model.py

The sparse topology is exported from organism O8. Effective edge weights
are short-term weight + long-term weight; both memory tiers remain available as
separate tensors for analysis. Axonal distance is converted to 60 Hz delay steps.
"""

import json
import torch
import torch.nn as nn
import snntorch as snn


# Auto-detect a Colab GPU (T4/L4) and fall back cleanly to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running AMMC-SNN on: {device}")

NEURON_IDS = ["N1","N2","N3","N4","N5","N6","N7","N8","N9"]
EDGE_LIST = [[0,4,1.19519653],[1,4,1.19519653],[0,1,0.57980556],[0,6,1.19759647],[1,2,0.61888116],[2,4,0.61018641],[2,5,0.58821925],[3,4,0.53389627],[3,6,0.59331601],[4,5,0.56157821],[4,6,1.19759647],[4,7,0.61684054],[5,7,0.59090495],[6,4,0.57067831],[7,4,0.59091973],[8,4,0.56324902],[8,6,0.57981309],[2,8,0.09078723],[2,1,0.05455766],[6,3,0.57330667],[6,7,0.61064613]]
SHORT_TERM_WEIGHTS = [0.60085664,0.58634068,0,0.63691339,0,0,0,0,0,0,0.6386837,0,0,0,0,0,0,0,0,0.493336,0.530668]
LONG_TERM_WEIGHTS = [0.59433989,0.60885585,0.57980556,0.56068307,0.61888116,0.61018641,0.58821925,0.53389627,0.59331601,0.56157821,0.55891277,0.61684054,0.59090495,0.57067831,0.59091973,0.56324902,0.57981309,0.09078723,0.05455766,0.07997067,0.07997813]
EDGE_SIGNS = [1,-1,1,1,-1,1,1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1,1,1]
EDGE_IDENTITIES = [{"edge_index":0,"source_id":"N1","target_id":"N5","dendrite_id":"N5:D3"},{"edge_index":1,"source_id":"N2","target_id":"N5","dendrite_id":"N5:D3"},{"edge_index":2,"source_id":"N1","target_id":"N2","dendrite_id":"N2:D4"},{"edge_index":3,"source_id":"N1","target_id":"N7","dendrite_id":"N7:D1"},{"edge_index":4,"source_id":"N2","target_id":"N3","dendrite_id":"N3:D3"},{"edge_index":5,"source_id":"N3","target_id":"N5","dendrite_id":"N5:D1"},{"edge_index":6,"source_id":"N3","target_id":"N6","dendrite_id":"N6:D4"},{"edge_index":7,"source_id":"N4","target_id":"N5","dendrite_id":"N5:D2"},{"edge_index":8,"source_id":"N4","target_id":"N7","dendrite_id":"N7:D4"},{"edge_index":9,"source_id":"N5","target_id":"N6","dendrite_id":"N6:D3"},{"edge_index":10,"source_id":"N5","target_id":"N7","dendrite_id":"N7:D1"},{"edge_index":11,"source_id":"N5","target_id":"N8","dendrite_id":"N8:D4"},{"edge_index":12,"source_id":"N6","target_id":"N8","dendrite_id":"N8:D1"},{"edge_index":13,"source_id":"N7","target_id":"N5","dendrite_id":"N5:D2"},{"edge_index":14,"source_id":"N8","target_id":"N5","dendrite_id":"N5:D1"},{"edge_index":15,"source_id":"N9","target_id":"N5","dendrite_id":"N5:D4"},{"edge_index":16,"source_id":"N9","target_id":"N7","dendrite_id":"N7:D3"},{"edge_index":17,"source_id":"N3","target_id":"N9","dendrite_id":"N9:D1"},{"edge_index":18,"source_id":"N3","target_id":"N2","dendrite_id":"N2:D3"},{"edge_index":19,"source_id":"N7","target_id":"N4","dendrite_id":"N4:D1"},{"edge_index":20,"source_id":"N7","target_id":"N8","dendrite_id":"N8:D3"}]
DELAY_MS = [665.5007,551.8168,393.8622,1044.073,428.1617,608.4395,602.9754,481.2965,476.0161,428.6328,583.2692,544.3732,408.2885,583.2692,544.3732,545.9104,1032.706,354.3369,428.1617,476.0161,487.3633]
DELAY_STEPS = [40,33,24,63,26,37,36,29,29,26,35,33,24,35,33,33,62,21,26,29,29]


class ExportedAMMC(nn.Module):
    """Sparse, recurrent LIF network translated from the AMMC-SNN connectome."""

    def __init__(self, beta=0.9):
        super().__init__()
        self.num_neurons = 9
        self.num_edges = 21
        self.neuron_ids = tuple(NEURON_IDS)
        self.neuron_index = {neuron_id: index for index, neuron_id in enumerate(NEURON_IDS)}
        self.edge_identities = tuple(EDGE_IDENTITIES)

        # edge_list columns: [source_index, target_index, STW + LTW]
        edge_list = torch.tensor(EDGE_LIST, dtype=torch.float32).reshape(-1, 3)
        self.register_buffer("edge_list", edge_list)
        self.register_buffer("edge_sources", edge_list[:, 0].long())
        self.register_buffer("edge_targets", edge_list[:, 1].long())
        # Trainable magnitudes. Synapse polarity remains fixed in edge_signs.
        self.edge_weights = nn.Parameter(edge_list[:, 2].clone())
        self.register_buffer(
            "short_term_weights",
            torch.tensor(SHORT_TERM_WEIGHTS, dtype=torch.float32),
        )
        self.register_buffer(
            "long_term_weights",
            torch.tensor(LONG_TERM_WEIGHTS, dtype=torch.float32),
        )
        self.register_buffer("edge_signs", torch.tensor(EDGE_SIGNS, dtype=torch.float32))
        self.register_buffer("delay_ms", torch.tensor(DELAY_MS, dtype=torch.float32))
        self.register_buffer("delays", torch.tensor(DELAY_STEPS, dtype=torch.long))

        self.lif = snn.Leaky(beta=beta)

    def save_to_json(self, path="colab_weights.json"):
        """Save trained edge magnitudes for atomic import into the JS sandbox."""
        trained_weights = self.edge_weights.detach().clamp(0.0, 1.0).cpu().tolist()
        edges = []
        for identity, weight in zip(self.edge_identities, trained_weights):
            edge_record = dict(identity)
            edge_record["weight"] = float(weight)
            edges.append(edge_record)

        payload = {
            "schema": "AMMC-SNN/colab-weights",
            "version": 1,
            "organism_id": "O8",
            "edges": edges,
        }
        with open(path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
        print(f"Saved {len(edges)} trained weights to {path}")
        return path

    def forward(self, x):
        """
        Args:
            x: External current/spike tensor shaped [time, batch, neurons].

        Returns:
            Output spikes shaped [time, batch, neurons].
        """
        if x.ndim != 3:
            raise ValueError("x must have shape [time, batch, neurons]")
        if x.size(-1) != self.num_neurons:
            raise ValueError(
                f"Expected {self.num_neurons} neurons, received {x.size(-1)}"
            )

        time_steps, batch_size, _ = x.shape
        membrane = torch.zeros(
            (batch_size, self.num_neurons), dtype=x.dtype, device=x.device
        )
        max_delay = int(self.delays.max().item()) if self.delays.numel() else 1
        delay_buffer = [
            torch.zeros(
                (batch_size, self.num_neurons), dtype=x.dtype, device=x.device
            )
            for _ in range(max_delay + 1)
        ]
        unique_delays = torch.unique(self.delays).detach().cpu().tolist()
        output_spikes = []

        for step in range(time_steps):
            slot = step % (max_delay + 1)
            current = x[step] + delay_buffer[slot]
            delay_buffer[slot] = torch.zeros_like(current)
            spikes, membrane = self.lif(current, membrane)
            output_spikes.append(spikes)

            if self.num_edges:
                source_spikes = spikes.index_select(1, self.edge_sources)
                edge_activity = source_spikes * (
                    self.edge_weights.clamp(0.0, 1.0) * self.edge_signs
                ).unsqueeze(0)

                # Group sparse edges by axonal delay, then accumulate at targets.
                for delay in unique_delays:
                    mask = self.delays == delay
                    arrival_slot = (step + int(delay)) % (max_delay + 1)
                    scheduled = torch.zeros_like(spikes).index_add(
                        1, self.edge_targets[mask], edge_activity[:, mask]
                    )
                    delay_buffer[arrival_slot] = delay_buffer[arrival_slot] + scheduled

        return torch.stack(output_spikes, dim=0)


if __name__ == "__main__":
    model = ExportedAMMC().to(device)
    # Mock input spike train: [time, batch, neurons]
    mock_input = torch.rand((10, 1, model.num_neurons), device=device)
    print("Testing forward pass...")
    output = model(mock_input)
    print("Execution successful! Output shape:", tuple(output.shape))
    print("After training, call model.save_to_json('colab_weights.json')")
# Actually call the function to generate the file
model.save_to_json('colab_weights.json')
print("Weights saved successfully!")