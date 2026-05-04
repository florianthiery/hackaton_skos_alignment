"""
gnn_pipeline_sketch.py
======================
Sketch of how to load the fetched JSKOS data into a graph and train a
heterogeneous GNN or hierarchical transformer for concept matching/alignment.

Install:
    pip install torch torch_geometric sentence-transformers pandas networkx
"""

import json
import pandas as pd
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Load edge list and assign integer node IDs
# ---------------------------------------------------------------------------

edges = pd.read_csv("data/edge_list.tsv", sep="\t", na_fill_value="")

# Collect all unique concept URIs as nodes
all_uris = pd.concat([edges["from_uri"], edges["to_uri"]]).unique()
uri_to_id = {uri: i for i, uri in enumerate(all_uris)}
print(f"Graph: {len(uri_to_id)} nodes, {len(edges)} edges")

# Map edge types to integers
edge_types = edges["mapping_type"].fillna("mappingRelation").unique()
type_to_id = {t: i for i, t in enumerate(edge_types)}
print(f"Edge types: {list(type_to_id.keys())}")

# Build edge index tensors (one per mapping type for heterogeneous GNN,
# or a single tensor with edge_attr for homogeneous GNN)
src = torch.tensor([uri_to_id[u] for u in edges["from_uri"]])
dst = torch.tensor([uri_to_id[u] for u in edges["to_uri"]])
edge_type = torch.tensor([type_to_id.get(t, 0) for t in edges["mapping_type"].fillna("")])


# ---------------------------------------------------------------------------
# 2. Build node features from concept labels
#    Option A: One-hot vocabulary membership (fast, sparse)
#    Option B: Sentence embeddings from prefLabel (richer, slower)
# ---------------------------------------------------------------------------

vocs = json.loads(Path("data/vocabularies.json").read_text())

# --- Option A: vocabulary membership as one-hot ---
voc_uris = [v["uri"] for v in vocs]
voc_to_id = {uri: i for i, uri in enumerate(voc_uris)}

# You'd need concept→voc mapping from the concept NDJSON files for full features.
# As a cheap proxy: match URI prefix to vocabulary namespace.
def guess_voc(concept_uri: str) -> int:
    for voc_uri, idx in voc_to_id.items():
        if concept_uri.startswith(voc_uri.rstrip("/").rstrip("#")):
            return idx
    return len(voc_uris)  # "unknown"

node_voc = torch.tensor([guess_voc(uri) for uri in all_uris])
# Embed as one-hot or pass through an nn.Embedding layer

# --- Option B: sentence embeddings (run once, cache) ---
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# Load concept labels from concepts/<slug>.ndjson and build uri→label dict
# labels = {uri: label_string, ...}
# embeddings = model.encode([labels.get(u, u) for u in all_uris], batch_size=256)
# node_features = torch.tensor(embeddings)


# ---------------------------------------------------------------------------
# 3a. Homogeneous GNN with typed edges
#     Suitable for: link prediction (will concept A match concept B?)
# ---------------------------------------------------------------------------

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv  # or GATConv, GCNConv

data = Data(
    num_nodes=len(uri_to_id),
    edge_index=torch.stack([src, dst]),
    edge_attr=edge_type,
    # x=node_features  # add once you have embeddings
)

# Make undirected (mappings often should be symmetric)
from torch_geometric.transforms import ToUndirected
data = ToUndirected()(data)

print(data)

class ConceptGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, n_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
        return self.convs[-1](x, edge_index)

# Link prediction head: dot product of node embeddings
class LinkPredictor(torch.nn.Module):
    def forward(self, z, src_idx, dst_idx):
        return (z[src_idx] * z[dst_idx]).sum(dim=-1).sigmoid()


# ---------------------------------------------------------------------------
# 3b. Heterogeneous GNN (recommended — preserves mapping type semantics)
#     Each mapping type is a distinct edge type in a HeteroData graph.
# ---------------------------------------------------------------------------

from torch_geometric.data import HeteroData

hetero = HeteroData()
hetero["concept"].num_nodes = len(uri_to_id)
# hetero["concept"].x = node_features

for mtype, tid in type_to_id.items():
    mask = edge_type == tid
    hetero["concept", mtype, "concept"].edge_index = torch.stack([
        src[mask], dst[mask]
    ])

print(hetero)

# Use torch_geometric.nn.HeteroConv wrapping SAGEConv or GATConv per relation.


# ---------------------------------------------------------------------------
# 4. Hierarchical Transformer approach
#    Treats each vocabulary as a "document", concepts within it as "tokens",
#    cross-vocabulary mappings become cross-document attention links.
# ---------------------------------------------------------------------------

# Rough sketch — not runnable without embeddings loaded:
#
# For each vocabulary V:
#   token_sequence = [concept_embeddings for concepts in V, sorted by notation/hierarchy]
#   Apply intra-vocabulary Transformer encoder → vocab_representation
#
# Then a second-level Transformer over vocabulary representations
# conditions on mapping edges to model cross-vocabulary alignment.
#
# This is essentially a Hierarchical Graph Transformer (HGT) pattern.
# See: torch_geometric.nn.HGTConv


# ---------------------------------------------------------------------------
# 5. Training signal options
# ---------------------------------------------------------------------------

# A) Self-supervised: predict masked mapping targets (like masked language modelling)
#    → mask 15% of edges, train model to reconstruct which concept they pointed to

# B) Supervised (if you have positive/negative labels from annotations):
#    annotations with bodyValue="+1" / "-1" are stored in the API
#    → fetch annotations, use as positive/negative training signal for link prediction

# C) Contrastive: concepts in exactMatch relations should have similar embeddings,
#    concepts in disjoint vocabularies without mappings should be dissimilar

print("Pipeline sketch loaded. Implement node features first, then pick option 3a or 3b.")