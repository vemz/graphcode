import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


NODE_TYPES = ["MODULE", "CLASS", "FUNCTION", "IMPORT"]
EDGE_TYPES = ["CONTAINS", "CALLS", "IMPORTS"]
TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}


def graph_to_pyg(record: dict) -> Data:
    nodes = record["nodes"]
    edges = record["edges"]
    node_ids = list(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    edge_src, edge_dst, edge_type = [], [], []
    for e in edges:
        if e["outNodeID"] in id_to_idx and e["inNodeID"] in id_to_idx:
            src = id_to_idx[e["outNodeID"]]
            dst = id_to_idx[e["inNodeID"]]
            edge_src.append(src)
            edge_dst.append(dst)
            edge_type.append(EDGE_TYPES.index(e["type"]) if e["type"] in EDGE_TYPES else 0)
    if not edge_src:
        edge_src = list(range(n))
        edge_dst = list(range(n))
        edge_type = [0] * n
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_index_undirected = to_undirected(edge_index)
    depth = {nid: 0 for nid in node_ids}
    contains_children = {nid: [] for nid in node_ids}
    for e in edges:
        if e["type"] == "CONTAINS" and e["outNodeID"] in nodes and e["inNodeID"] in nodes:
            contains_children[e["outNodeID"]].append(e["inNodeID"])
    roots = [nid for nid, nd in nodes.items() if nd["type"] == "MODULE"]
    visited = set()
    queue = [(r, 0) for r in roots]
    while queue:
        nid, d = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        depth[nid] = d
        for child in contains_children.get(nid, []):
            if child not in visited:
                queue.append((child, d + 1))
    in_deg = [0] * n
    out_deg = [0] * n
    contains_inc = [0] * n
    calls_inc = [0] * n
    imports_inc = [0] * n
    for e in edges:
        if e["outNodeID"] not in id_to_idx or e["inNodeID"] not in id_to_idx:
            continue
        s, d = id_to_idx[e["outNodeID"]], id_to_idx[e["inNodeID"]]
        out_deg[s] += 1
        in_deg[d] += 1
        if e["type"] == "CONTAINS":
            contains_inc[s] += 1
            contains_inc[d] += 1
        elif e["type"] == "CALLS":
            calls_inc[s] += 1
            calls_inc[d] += 1
        elif e["type"] == "IMPORTS":
            imports_inc[s] += 1
            imports_inc[d] += 1
    x = torch.zeros(n, 16, dtype=torch.float32)
    y = torch.zeros(n, dtype=torch.long)
    for i, nid in enumerate(node_ids):
        nd = nodes[nid]
        ntype = nd["type"]
        type_idx = TYPE_TO_IDX.get(ntype, 0)
        x[i, type_idx] = 1.0
        y[i] = type_idx
        code = nd.get("code", "")
        x[i, 4] = math.log1p(len(code)) / 10.0
        x[i, 5] = math.log1p(code.count("\n")) / 5.0
        x[i, 6] = depth[nid] / 5.0
        x[i, 7] = math.log1p(in_deg[i]) / 3.0
        x[i, 8] = math.log1p(out_deg[i]) / 3.0
        total_inc = contains_inc[i] + calls_inc[i] + imports_inc[i] + 1e-6
        x[i, 9] = contains_inc[i] / total_inc
        x[i, 10] = calls_inc[i] / total_inc
        x[i, 11] = imports_inc[i] / total_inc
    data = Data(
        x=x,
        edge_index=edge_index_undirected,
        y=y,
        num_nodes=n,
    )
    return data


def load_pyg_dataset(jsonl_path: Path) -> List[Data]:
    graphs = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            try:
                data = graph_to_pyg(record)
                if data.num_nodes >= 3:
                    graphs.append(data)
            except Exception as e:
                print(f"Skip graph: {e}")
    return graphs


def compute_low_freq_contributions(edge_index: torch.Tensor, num_nodes: int, K: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    if num_nodes < 3:
        return torch.ones(num_nodes), torch.ones(edge_index.size(1))
    A = torch.zeros(num_nodes, num_nodes)
    for i in range(edge_index.size(1)):
        s, d = edge_index[0, i].item(), edge_index[1, i].item()
        if s != d:
            A[s, d] = 1.0
            A[d, s] = 1.0
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    L = torch.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
    try:
        eigvals, eigvecs = torch.linalg.eigh(L)
    except Exception:
        return torch.ones(num_nodes), torch.ones(edge_index.size(1))
    K_eff = min(K, num_nodes)
    eigvals = eigvals[:K_eff]
    eigvecs = eigvecs[:, :K_eff]
    C_E_per_edge = torch.zeros(edge_index.size(1))
    for idx in range(edge_index.size(1)):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        contribs = torch.abs(eigvecs[i] * eigvals * eigvecs[j])
        denom = contribs.sum() + 1e-8
        cumsum = torch.cumsum(contribs, dim=0)
        C_E_per_edge[idx] = (cumsum / denom).mean()
    C_N = torch.zeros(num_nodes)
    counts = torch.zeros(num_nodes)
    for idx in range(edge_index.size(1)):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        C_N[i] += C_E_per_edge[idx]
        C_N[j] += C_E_per_edge[idx]
        counts[i] += 1
        counts[j] += 1
    C_N = C_N / (counts + 1e-8)
    return C_N, C_E_per_edge


def precompute_contributions(graphs: List[Data], K: int = 8) -> List[Data]:
    print(f"Pré-calcul des contributions spectrales (K={K})...")
    for data in tqdm(graphs):
        C_N, C_E = compute_low_freq_contributions(data.edge_index, data.num_nodes, K=K)
        data.c_n = C_N
        data.c_e = C_E
    return graphs


def corrupt_graph(data: Data, rate_n: float = 0.3, rate_e: float = 0.3) -> Data:
    n = data.num_nodes
    n_mask = max(1, int(rate_n * n))
    n_drop_edges = max(1, int(rate_e * data.edge_index.size(1)))
    probs_n = data.c_n + 1e-6
    probs_n = probs_n / probs_n.sum()
    masked_nodes_value = torch.multinomial(probs_n, n_mask, replacement=False)
    ranks_n = torch.argsort(data.c_n, descending=True).float() + 1.0
    probs_rank = (1.0 / ranks_n)
    probs_rank = probs_rank / probs_rank.sum()
    masked_nodes_rank = torch.multinomial(probs_rank, n_mask, replacement=False)
    masked_nodes = torch.unique(torch.cat([masked_nodes_value, masked_nodes_rank]))
    x_corrupt = data.x.clone()
    mask_vec = torch.zeros(n, dtype=torch.bool)
    mask_vec[masked_nodes] = True
    probs_e = data.c_e + 1e-6
    probs_e = probs_e / probs_e.sum()
    drop_edges = torch.multinomial(probs_e, min(n_drop_edges, data.edge_index.size(1)), replacement=False)
    keep_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    keep_mask[drop_edges] = False
    edge_index_corrupt = data.edge_index[:, keep_mask]
    return Data(
        x=x_corrupt,
        edge_index=edge_index_corrupt,
        y=data.y,
        num_nodes=n,
        node_mask=mask_vec,
    )


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64, heads: int = 4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(in_dim))
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=0.1)

    def forward(self, x, edge_index, node_mask=None):
        if node_mask is not None:
            x = torch.where(node_mask.unsqueeze(-1), self.mask_token.unsqueeze(0).expand_as(x), x)
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.conv2(h, edge_index)
        return h


class FCGSSL(nn.Module):
    def __init__(self, in_dim: int = 16, hidden_dim: int = 64, embed_dim: int = 64):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index, node_mask=getattr(data, "node_mask", None))
        x_recon = self.decoder(z)
        return z, x_recon


def sce_loss(x_recon: torch.Tensor, x_target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    x_recon_n = F.normalize(x_recon, dim=-1)
    x_target_n = F.normalize(x_target, dim=-1)
    cos = (x_recon_n * x_target_n).sum(dim=-1)
    return (1 - cos).pow(gamma).mean()


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def train(graphs: List[Data], epochs: int = 50, lr: float = 1e-3,
          batch_size: int = 16, device: str = "cpu") -> FCGSSL:
    print(f"Training FC-GSSL on {len(graphs)} graphs, device={device}")
    model = FCGSSL(in_dim=16, hidden_dim=64, embed_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        model.train()
        for batch in loader:
            batch = batch.to(device)
            view1_list = [corrupt_graph(g.cpu(), rate_n=0.3, rate_e=0.1) for g in batch.to_data_list()]
            view1 = Batch.from_data_list(view1_list).to(device)
            view2_list = [corrupt_graph(g.cpu(), rate_n=0.1, rate_e=0.3) for g in batch.to_data_list()]
            view2 = Batch.from_data_list(view2_list).to(device)
            z1, x_recon1 = model(view1)
            z2, x_recon2 = model(view2)
            mask1 = view1.node_mask
            loss_recon = sce_loss(x_recon1[mask1], batch.x[mask1]) if mask1.any() else torch.tensor(0.0, device=device)
            loss_align = info_nce(z1, z2)
            loss = loss_recon + 0.1 * loss_align
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={total_loss/n_batches:.4f}")
    return model


def evaluate_downstream(model: FCGSSL, graphs: List[Data], device: str = "cpu") -> Dict:
    print("\nÉvaluation downstream (classification du type de nœud)...")
    model.eval()
    all_z, all_y = [], []
    with torch.no_grad():
        loader = DataLoader(graphs, batch_size=16, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            z = model.encoder(batch.x, batch.edge_index)
            all_z.append(z.cpu().numpy())
            all_y.append(batch.y.cpu().numpy())
    Z = np.concatenate(all_z)
    Y = np.concatenate(all_y)
    X_raw_list = []
    for g in graphs:
        X_raw_list.append(g.x.numpy())
    X_raw = np.concatenate(X_raw_list)
    X_train_emb, X_test_emb, y_train, y_test = train_test_split(Z, Y, test_size=0.2, random_state=42, stratify=Y)
    X_train_raw, X_test_raw, _, _ = train_test_split(X_raw, Y, test_size=0.2, random_state=42, stratify=Y)
    clf_emb = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf_emb.fit(X_train_emb, y_train)
    pred_emb = clf_emb.predict(X_test_emb)
    acc_emb = accuracy_score(y_test, pred_emb)
    f1_emb = f1_score(y_test, pred_emb, average="macro")
    clf_raw = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf_raw.fit(X_train_raw, y_train)
    pred_raw = clf_raw.predict(X_test_raw)
    acc_raw = accuracy_score(y_test, pred_raw)
    f1_raw = f1_score(y_test, pred_raw, average="macro")
    results = {
        "embeddings": {"accuracy": acc_emb, "macro_f1": f1_emb},
        "raw_features": {"accuracy": acc_raw, "macro_f1": f1_raw},
        "improvement_accuracy": acc_emb - acc_raw,
        "improvement_f1": f1_emb - f1_raw,
        "n_test_nodes": len(y_test),
    }
    print(f"  Baseline (features brutes)   : acc={acc_raw:.4f} | f1={f1_raw:.4f}")
    print(f"  FC-GSSL embeddings           : acc={acc_emb:.4f} | f1={f1_emb:.4f}")
    print(f"  Δ accuracy                   : {acc_emb - acc_raw:+.4f}")
    print(f"  Δ macro-f1                   : {f1_emb - f1_raw:+.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, default="poc_dataset/graphs.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=8, help="Top-K eigenvectors for spectral analysis")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu / mps (Apple Silicon) / cuda")
    args = parser.parse_args()
    print(f"Chargement de {args.graphs}...")
    graphs = load_pyg_dataset(Path(args.graphs))
    print(f"  {len(graphs)} graphes valides")
    print(f"  Total nœuds : {sum(g.num_nodes for g in graphs)}")
    graphs = precompute_contributions(graphs, K=args.K)
    model = train(graphs, epochs=args.epochs, lr=args.lr,
                  batch_size=args.batch_size, device=args.device)
    results = evaluate_downstream(model, graphs, device=args.device)
    out = Path("poc_dataset/fc_gssl_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats sauvegardés : {out}")
    torch.save(model.state_dict(), "poc_dataset/fc_gssl_model.pt")
    print("Modèle sauvegardé : poc_dataset/fc_gssl_model.pt")