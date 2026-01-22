# visualize_omniglot_graph.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# --- Paths ---
monty_models_dir = os.getenv("MONTY_MODELS", "~/tbp/models")
model_path = Path(monty_models_dir).expanduser() / "omniglot" / "omniglot_training" / "pretrained" / "model.pt"
output_dir = Path("~/tbp/results/monty/visualizations/omniglot").expanduser()

# --- Check model ---
if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}\n"
                            "Run: python benchmarks/run.py -e omniglot_training")

output_dir.mkdir(parents=True, exist_ok=True)

# --- Load model ---
print(f"✅ Loading model from {model_path}")
ckpt = torch.load(model_path, map_location="cpu")

# --- Extract processed observations ---
sm_dict = ckpt.get("sm_dict", {})
if not sm_dict or 0 not in sm_dict:
    raise KeyError("No sm_dict or sensor module 0 in checkpoint")

observations = sm_dict[0].get("processed_observations", [])
if not observations:
    raise ValueError("No processed_observations in sm_dict[0]")

print(f"🔍 Found {len(observations)} processed observations")

# --- Helper: Deep location extractor ---
def extract_location(obs):
    # Try direct .location
    if hasattr(obs, "location") and obs.location is not None:
        loc = obs.location
        if torch.is_tensor(loc):
            loc = loc.cpu().numpy()
        return np.array(loc[:2])  # x, y only

    # Try .morphological_features.location
    if hasattr(obs, "morphological_features"):
        mf = obs.morphological_features
        if hasattr(mf, "location") and mf.location is not None:
            loc = mf.location
            if torch.is_tensor(loc):
                loc = loc.cpu().numpy()
            return np.array(loc[:2])

    # Try .features.location (common in EvidenceGraphLM)
    if hasattr(obs, "features") and hasattr(obs.features, "location"):
        loc = obs.features.location
        if torch.is_tensor(loc):
            loc = loc.cpu().numpy()
        return np.array(loc[:2])

    # Try __dict__ traversal
    if hasattr(obs, "__dict__"):
        for k, v in obs.__dict__.items():
            if "location" in k.lower() and v is not None:
                if torch.is_tensor(v):
                    v = v.cpu().numpy()
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                    return np.array(v[:2])

    # Try as dict
    if isinstance(obs, dict):
        for k, v in obs.items():
            if "location" in k.lower() and v is not None:
                if torch.is_tensor(v):
                    v = v.cpu().numpy()
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                    return np.array(v[:2])

    return None

# --- Extract locations ---
locations = []
for i, obs in enumerate(observations):
    loc = extract_location(obs)
    if loc is not None:
        locations.append(loc)
    else:
        print(f"⚠️  No location in observation {i} (type: {type(obs)})")

print(f"🎯 Extracted {len(locations)} locations from observations")

if len(locations) == 0:
    raise ValueError("No valid locations extracted from observations")

# --- Build Sequential Path Graph ---
G = nx.DiGraph()

# Add nodes in order
for i, (x, y) in enumerate(locations):
    G.add_node(i, pos=(x, -y), label=f"{i}\n({x:.0f},{y:.0f})")  # Flip y for upright

# Add edges: step i → i+1
for i in range(len(locations) - 1):
    G.add_edge(i, i + 1)

# --- Plot ---
plt.figure(figsize=(12, 10))
pos = nx.get_node_attributes(G, 'pos')
labels = nx.get_node_attributes(G, 'label')

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color="gray", width=2.0, arrows=True, arrowsize=15)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=400, node_color="lightblue", edgecolors="black")

# Draw labels
nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight="bold")

plt.title("Monty's Sensing Path on Omniglot Character\n(Sequential Stroke Following)", fontsize=16)
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save and show
plot_path = output_dir / "omniglot_sensing_path.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()

print(f"🖼️  Saved sensing path: {plot_path}")

# --- Save as JSON ---
path_data = {
    "character": "omniglot_trained",
    "num_steps": len(locations),
    "locations": [{"step": i, "x": float(x), "y": float(y)} for i, (x, y) in enumerate(locations)],
    "edges": [[i, i+1] for i in range(len(locations)-1)]
}
with open(output_dir / "sensing_path.json", "w") as f:
    import json
    json.dump(path_data, f, indent=2)

print(f"✅ Full analysis saved to: {output_dir}")