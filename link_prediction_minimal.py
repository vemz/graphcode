import argparse
import json
import math
import random
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


NODE_TYPES = ["MODULE", "CLASS", "FUNCTION", "IMPORT"]
EDGE_TYPES = ["CONTAINS", "CALLS", "IMPORTS"]
TYPE_TO_IDX = {t: i for i, t in enumerate(NODE_TYPES)}
FEATURE_DIM = 8  # Réduit de 16 à 8

def graph_to_pyg_minimal(record: dict) -> Tuple[Data, List[Tuple[int, int]]]:
    nodes = record["nodes"]
    edges = record["edges"]

    node_ids = list(nodes.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    structural_edges = []
    calls_edges = []
    for e in edges:
        if e["outNodeID"] not in id_to_idx or e["inNodeID"] not in id_to_idx:
            continue
        s, d = id_to_idx[e["outNodeID"]], id_to_idx[e["inNodeID"]]
        if e["type"] == "CALLS":
            calls_edges.append((s, d))
        else:
            structural_edges.append((s, d))

    if structural_edges:
        edge_src = [e[0] for e in structural_edges]
        edge_dst = [e[1] for e in structural_edges]
    else:
        edge_src = list(range(n))
        edge_dst = list(range(n))

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)

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

    x = torch.zeros(n, FEATURE_DIM, dtype=torch.float32)
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

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=n)
    return data, calls_edges

def generate_link_pairs(graphs_with_calls, seed: int = 42) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    all_pos, all_neg = [], []
    for g_idx, (data, calls) in enumerate(graphs_with_calls):
        n = data.num_nodes
        if n < 4 or len(calls) == 0:
            continue
        for s, d in calls:
            all_pos.append((g_idx, s, d))
        calls_set = set(calls)
        n_neg_needed = len(calls)
        attempts = 0
        neg_count = 0
        while neg_count < n_neg_needed and attempts < n_neg_needed * 10:
            s = random.randint(0, n - 1)
            d = random.randint(0, n - 1)
            if s != d and (s, d) not in calls_set:
                all_neg.append((g_idx, s, d))
                neg_count += 1
            attempts += 1

    print(f"  Total positifs : {len(all_pos)}")
    print(f"  Total négatifs : {len(all_neg)}")

    all_pos = np.array(all_pos)
    all_neg = np.array(all_neg)
    np.random.shuffle(all_pos)
    np.random.shuffle(all_neg)

    n_pos, n_neg = len(all_pos), len(all_neg)
    n_train_pos, n_val_pos = int(0.8 * n_pos), int(0.1 * n_pos)
    n_train_neg, n_val_neg = int(0.8 * n_neg), int(0.1 * n_neg)

    splits = {
        "train": {
            "pairs": np.concatenate([all_pos[:n_train_pos], all_neg[:n_train_neg]]),
            "labels": np.concatenate([np.ones(n_train_pos), np.zeros(n_train_neg)]),
        },
        "val": {
            "pairs": np.concatenate([all_pos[n_train_pos:n_train_pos + n_val_pos],
                                      all_neg[n_train_neg:n_train_neg + n_val_neg]]),
            "labels": np.concatenate([np.ones(n_val_pos), np.zeros(n_val_neg)]),
        },
        "test": {
            "pairs": np.concatenate([all_pos[n_train_pos + n_val_pos:],
                                      all_neg[n_train_neg + n_val_neg:]]),
            "labels": np.concatenate([np.ones(n_pos - n_train_pos - n_val_pos),
                                       np.zeros(n_neg - n_train_neg - n_val_neg)]),
        },
    }
    for split_name, split_data in splits.items():
        perm = np.random.permutation(len(split_data["pairs"]))
        split_data["pairs"] = split_data["pairs"][perm]
        split_data["labels"] = split_data["labels"][perm]
        print(f"  {split_name}: {len(split_data['pairs'])} pairs ({int(split_data['labels'].sum())} pos)")
    return splits

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
    def __init__(self, in_dim: int = FEATURE_DIM, hidden_dim: int = 64, embed_dim: int = 64):
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


def compute_low_freq_contributions(edge_index, num_nodes, K=8):
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
    C_E = torch.zeros(edge_index.size(1))
    for idx in range(edge_index.size(1)):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        contribs = torch.abs(eigvecs[i] * eigvals * eigvecs[j])
        denom = contribs.sum() + 1e-8
        cumsum = torch.cumsum(contribs, dim=0)
        C_E[idx] = (cumsum / denom).mean()
    C_N = torch.zeros(num_nodes)
    counts = torch.zeros(num_nodes)
    for idx in range(edge_index.size(1)):
        i, j = edge_index[0, idx].item(), edge_index[1, idx].item()
        C_N[i] += C_E[idx]
        C_N[j] += C_E[idx]
        counts[i] += 1
        counts[j] += 1
    C_N = C_N / (counts + 1e-8)
    return C_N, C_E


def corrupt_graph(data, rate_n=0.3, rate_e=0.3):
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

    return Data(x=x_corrupt, edge_index=edge_index_corrupt, y=data.y, num_nodes=n, node_mask=mask_vec)


def sce_loss(x_recon, x_target, gamma=2.0):
    x_recon_n = F.normalize(x_recon, dim=-1)
    x_target_n = F.normalize(x_target, dim=-1)
    cos = (x_recon_n * x_target_n).sum(dim=-1)
    return (1 - cos).pow(gamma).mean()


def info_nce(z1, z2, tau=0.2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / tau
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)


def pretrain_fc_gssl(graphs, epochs=50, lr=1e-3, batch_size=16, device="cpu"):
    for data in tqdm(graphs):
        C_N, C_E = compute_low_freq_contributions(data.edge_index, data.num_nodes, K=8)
        data.c_n = C_N
        data.c_e = C_E

    model = FCGSSL(in_dim=FEATURE_DIM, hidden_dim=64, embed_dim=64).to(device)
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

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | loss={total_loss/n_batches:.4f}")
    return model

def get_node_embeddings(model, graphs, device="cpu"):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in graphs:
            data = data.to(device)
            z = model.encoder(data.x, data.edge_index)
            embeddings.append(z.cpu().numpy())
    return embeddings


def build_pair_features(splits, graphs, node_embeddings):
    features = {}
    for split_name, split_data in splits.items():
        pairs = split_data["pairs"]
        raw_concat, raw_dot, emb_concat = [], [], []
        for g_idx, src, dst in pairs:
            g_idx, src, dst = int(g_idx), int(src), int(dst)
            x_src = graphs[g_idx].x[src].numpy()
            x_dst = graphs[g_idx].x[dst].numpy()
            raw_concat.append(np.concatenate([x_src, x_dst]))
            raw_dot.append([float(np.dot(x_src, x_dst))])

            z_src = node_embeddings[g_idx][src]
            z_dst = node_embeddings[g_idx][dst]
            emb_concat.append(np.concatenate([z_src, z_dst, z_src * z_dst, np.abs(z_src - z_dst)]))

        features[split_name] = {
            "raw_concat": np.array(raw_concat),
            "raw_dot": np.array(raw_dot),
            "emb_concat": np.array(emb_concat),
            "labels": split_data["labels"],
        }
    return features


def train_and_eval_classifier(features, feature_key, name):
    X_train = features["train"][feature_key]
    y_train = features["train"]["labels"]
    X_val = features["val"][feature_key]
    y_val = features["val"]["labels"]
    X_test = features["test"][feature_key]
    y_test = features["test"]["labels"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                        random_state=42, early_stopping=True, validation_fraction=0.1)
    clf.fit(X_train_s, y_train)

    val_scores = clf.predict_proba(X_val_s)[:, 1]
    test_scores = clf.predict_proba(X_test_s)[:, 1]
    val_pred = clf.predict(X_val_s)
    test_pred = clf.predict(X_test_s)

    results = {
        "val_auc": roc_auc_score(y_val, val_scores),
        "val_ap": average_precision_score(y_val, val_scores),
        "val_acc": (val_pred == y_val).mean(),
        "test_auc": roc_auc_score(y_test, test_scores),
        "test_ap": average_precision_score(y_test, test_scores),
        "test_acc": (test_pred == y_test).mean(),
    }
    print(f"\n  [{name}]")
    print(f"    Val  : AUC={results['val_auc']:.4f} | AP={results['val_ap']:.4f} | Acc={results['val_acc']:.4f}")
    print(f"    Test : AUC={results['test_auc']:.4f} | AP={results['test_ap']:.4f} | Acc={results['test_acc']:.4f}")
    return results


def dot_product_baseline(features):
    test_scores = features["test"]["raw_dot"].flatten()
    test_labels = features["test"]["labels"]
    val_scores = features["val"]["raw_dot"].flatten()
    val_labels = features["val"]["labels"]
    results = {
        "val_auc": roc_auc_score(val_labels, val_scores),
        "val_ap": average_precision_score(val_labels, val_scores),
        "test_auc": roc_auc_score(test_labels, test_scores),
        "test_ap": average_precision_score(test_labels, test_scores),
    }
    print(f"\n  [Baseline dot-product (features brutes, pas d'apprentissage)]")
    print(f"    Val  : AUC={results['val_auc']:.4f} | AP={results['val_ap']:.4f}")
    print(f"    Test : AUC={results['test_auc']:.4f} | AP={results['test_ap']:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=str, default="poc_dataset/graphs.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="poc_dataset/link_pred_minimal_results.json")
    args = parser.parse_args()

    print(f"Chargement de {args.graphs} (features MINIMALES, dim={FEATURE_DIM})...")
    graphs_with_calls = []
    with open(args.graphs) as f:
        for line in f:
            record = json.loads(line)
            try:
                data, calls = graph_to_pyg_minimal(record)
                if data.num_nodes >= 4:
                    graphs_with_calls.append((data, calls))
            except Exception:
                continue
    print(f"  {len(graphs_with_calls)} graphes chargés")
    total_calls = sum(len(c) for _, c in graphs_with_calls)
    print(f"  {total_calls} arêtes CALLS au total")

    print("\nGénération des paires train/val/test...")
    splits = generate_link_pairs(graphs_with_calls)

    graphs_pyg = [data for data, _ in graphs_with_calls]
    print("\nPré-entraînement FC-GSSL avec features minimales...")
    model = pretrain_fc_gssl(graphs_pyg, epochs=args.epochs, lr=args.lr,
                              batch_size=args.batch_size, device=args.device)

    print("\nCalcul des embeddings finaux...")
    node_embeddings = get_node_embeddings(model, graphs_pyg, device=args.device)

    print("Construction des features de paires...")
    features = build_pair_features(splits, graphs_pyg, node_embeddings)

    print("\n" + "=" * 60)
    print("RÉSULTATS LINK PREDICTION (features brutes MINIMALES)")
    print("=" * 60)

    all_results = {}
    all_results["baseline_dot"] = dot_product_baseline(features)
    all_results["baseline_mlp_raw"] = train_and_eval_classifier(features, "raw_concat", "MLP sur features brutes concat (8+8=16 dims)")
    all_results["fc_gssl_mlp"] = train_and_eval_classifier(features, "emb_concat", "MLP sur embeddings FC-GSSL (256 dims)")

    print("\n" + "=" * 60)
    print("RÉSUMÉ COMPARATIF (Test set)")
    print("=" * 60)
    print(f"  Baseline dot-product       : AUC = {all_results['baseline_dot']['test_auc']:.4f}")
    print(f"  Baseline MLP features brutes : AUC = {all_results['baseline_mlp_raw']['test_auc']:.4f}")
    print(f"  FC-GSSL MLP embeddings      : AUC = {all_results['fc_gssl_mlp']['test_auc']:.4f}")
    delta = all_results['fc_gssl_mlp']['test_auc'] - all_results['baseline_mlp_raw']['test_auc']
    print(f"\n  Δ FC-GSSL vs Baseline MLP   : {delta:+.4f}")

    # Calcul de la réduction d'erreurs relative
    err_baseline = 1 - all_results['baseline_mlp_raw']['test_acc']
    err_fcgssl = 1 - all_results['fc_gssl_mlp']['test_acc']
    if err_baseline > 0:
        rel_reduction = (err_baseline - err_fcgssl) / err_baseline
        print(f"  Réduction relative erreurs  : {rel_reduction*100:+.1f}%")

    if delta > 0.10:
        print("\n  ✅✅ FC-GSSL apporte une amélioration TRÈS significative.")
    elif delta > 0.05:
        print("\n  ✅ FC-GSSL apporte une amélioration significative.")
    elif delta > 0.01:
        print("\n  🟡 FC-GSSL apporte une amélioration modeste.")
    else:
        print("\n  ❌ FC-GSSL n'améliore pas significativement la prédiction.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRésultats sauvegardés : {args.output}")


if __name__ == "__main__":
    main()