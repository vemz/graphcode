import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

def generate_synthetic_graph(rng: random.Random, max_nodes: int = 30) -> Dict:
    nodes: Dict[str, dict] = {}
    edges: List[dict] = []

    module_id = "module_synth"
    nodes[module_id] = {"id": module_id, "name": "synth_module", "type": "MODULE"}

    n_imports = rng.randint(0, 6) 
    for i in range(n_imports):
        if len(nodes) >= max_nodes:
            break
        iid = f"import_I{i}"
        nodes[iid] = {"id": iid, "name": f"library_{i}", "type": "IMPORT"}
        edges.append({"outNodeID": module_id, "inNodeID": iid, "type": "IMPORTS"})

    n_classes = rng.randint(0, 3)
    class_ids = []
    for k in range(n_classes):
        if len(nodes) >= max_nodes:
            break
        cid = f"class_C{k}"
        nodes[cid] = {"id": cid, "name": f"Class{k}", "type": "CLASS"}
        edges.append({"outNodeID": module_id, "inNodeID": cid, "type": "CONTAINS"})
        class_ids.append(cid)

        n_methods = rng.randint(1, 4)
        for m in range(n_methods):
            if len(nodes) >= max_nodes:
                break
            fid = f"func_C{k}_m{m}"
            nodes[fid] = {"id": fid, "name": f"method_{m}", "type": "FUNCTION", "scope": cid}
            edges.append({"outNodeID": cid, "inNodeID": fid, "type": "CONTAINS"})

    n_top_funcs = rng.randint(1, 4)
    for l in range(n_top_funcs):
        if len(nodes) >= max_nodes:
            break
        fid = f"func_top_{l}"
        nodes[fid] = {"id": fid, "name": f"top_func_{l}", "type": "FUNCTION", "scope": module_id}
        edges.append({"outNodeID": module_id, "inNodeID": fid, "type": "CONTAINS"})

    func_nodes = [nid for nid, nd in nodes.items() if nd["type"] == "FUNCTION"]
    func_idx = {fid: i for i, fid in enumerate(func_nodes)}

    if len(func_nodes) >= 2:
        for src in func_nodes:
            if rng.random() < 0.5:
                continue

            src_scope = nodes[src].get("scope", module_id)
            n_calls = rng.randint(0, 2)
            if n_calls == 0:
                continue

            candidates = []
            for tgt in func_nodes:
                if tgt == src:
                    continue
                if func_idx[tgt] >= func_idx[src]:
                    continue
                tgt_scope = nodes[tgt].get("scope", module_id)

                if tgt_scope == src_scope and src_scope != module_id:
                    weight = 6.0 
                elif tgt_scope == module_id:
                    weight = 4.0  
                else:
                    weight = 1.0  

                candidates.append((tgt, weight))

            if not candidates:
                continue

            tgts = [c[0] for c in candidates]
            weights = np.array([c[1] for c in candidates], dtype=np.float64)
            weights = weights / weights.sum()
            n_actual = min(n_calls, len(tgts))
            chosen_idx = np.random.choice(len(tgts), size=n_actual, replace=False, p=weights)
            for ci in chosen_idx:
                edges.append({"outNodeID": src, "inNodeID": tgts[ci], "type": "CALLS"})

    for nd in nodes.values():
        nd.pop("scope", None)

    type_counts = Counter(nd["type"] for nd in nodes.values())
    edge_type_counts = Counter(e["type"] for e in edges)
    degree = Counter()
    for e in edges:
        degree[e["outNodeID"]] += 1
        degree[e["inNodeID"]] += 1
    n = len(nodes)
    avg_deg = sum(degree.values()) / n if n else 0
    isolated = sum(1 for nid in nodes if degree[nid] == 0)

    stats = {
        "n_nodes": n,
        "n_edges": len(edges),
        "avg_degree": round(avg_deg, 2),
        "isolated_nodes": isolated,
        "node_types": dict(type_counts),
        "edge_types": dict(edge_type_counts),
    }

    return {"nodes": nodes, "edges": edges, "stats": stats}

def generate_dataset(n_graphs: int, output_path: Path, seed: int = 42,
                     max_nodes: int = 30, require_call: bool = False):
    rng = random.Random(seed)
    np.random.seed(seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_filtered = 0
    global_stats = {
        "total_nodes": 0,
        "total_edges": 0,
        "node_types": Counter(),
        "edge_types": Counter(),
        "size_distribution": Counter(),
        "calls_count": [],
    }

    with open(output_path, "w") as f:
        for i in tqdm(range(n_graphs), desc="Generating"):
            graph = generate_synthetic_graph(rng, max_nodes=max_nodes)
            stats = graph["stats"]

            if stats["n_nodes"] < 3:
                n_filtered += 1
                continue
            if require_call and stats["edge_types"].get("CALLS", 0) < 1:
                n_filtered += 1
                continue

            record = {
                "repo": "synthetic",
                "file": f"graph_{i}.py",
                "nodes": graph["nodes"],
                "edges": graph["edges"],
                "stats": stats,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

            global_stats["total_nodes"] += stats["n_nodes"]
            global_stats["total_edges"] += stats["n_edges"]
            global_stats["node_types"].update(stats["node_types"])
            global_stats["edge_types"].update(stats["edge_types"])
            global_stats["size_distribution"][stats["n_nodes"]] += 1
            global_stats["calls_count"].append(stats["edge_types"].get("CALLS", 0))

    print("\n" + "=" * 60)
    print("DATASET SYNTHÉTIQUE v2 GÉNÉRÉ")
    print("=" * 60)
    print(f"  Graphes valides : {n_ok}")
    print(f"  Filtrés         : {n_filtered}")
    print(f"\n  Nœuds moy. / graphe  : {global_stats['total_nodes'] / n_ok:.1f}")
    print(f"  Arêtes moy. / graphe : {global_stats['total_edges'] / n_ok:.1f}")
    print(f"  CALLS moy. / graphe  : {np.mean(global_stats['calls_count']):.1f}")
    print(f"  CALLS médian / graphe: {np.median(global_stats['calls_count']):.1f}")
    print(f"  Graphes sans CALLS   : {sum(1 for c in global_stats['calls_count'] if c == 0)} "
          f"({sum(1 for c in global_stats['calls_count'] if c == 0)/n_ok*100:.1f}%)")
    print(f"\n  Types de nœuds : {dict(global_stats['node_types'])}")
    print(f"  Types d'arêtes : {dict(global_stats['edge_types'])}")
    print(f"\n  Distribution des tailles (5 plus fréquentes) :")
    for size, count in global_stats["size_distribution"].most_common(5):
        print(f"    n={size}: {count} graphes")
    print(f"\n  Output : {output_path}")


def compare_with_real(synthetic_path: Path, real_path: Path = None):
    print("\n" + "=" * 60)
    print("COMPARAISON SYNTHÉTIQUE v2 vs RÉEL")
    print("=" * 60)

    def load_stats(path):
        ns, calls_ratio, types = [], [], Counter()
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                s = r["stats"]
                ns.append(s["n_nodes"])
                total_edges = s["n_edges"]
                if total_edges > 0:
                    calls_ratio.append(s["edge_types"].get("CALLS", 0) / total_edges)
                types.update(s["node_types"])
        return ns, calls_ratio, types

    s_ns, s_cr, s_types = load_stats(synthetic_path)
    print(f"\n  [Synthétique v2]")
    print(f"    n_nodes  : moy={np.mean(s_ns):.1f}, med={np.median(s_ns):.0f}, min={min(s_ns)}, max={max(s_ns)}")
    print(f"    CALLS%   : moy={np.mean(s_cr)*100:.1f}%, med={np.median(s_cr)*100:.1f}%")
    total_s = sum(s_types.values())
    print(f"    Types    : " + ", ".join(f"{t}={v/total_s*100:.1f}%" for t, v in s_types.items()))

    if real_path and Path(real_path).exists():
        r_ns, r_cr, r_types = load_stats(real_path)
        # Filtre le réel à n <= max(synth) pour comparaison équitable
        r_ns_filt = [n for n in r_ns if n <= 30]
        print(f"\n  [Réel filtré n<=30 ({real_path.name})]")
        print(f"    n_graphs : {len(r_ns_filt)} / {len(r_ns)} total ({len(r_ns_filt)/len(r_ns)*100:.1f}%)")
        print(f"    n_nodes  : moy={np.mean(r_ns_filt):.1f}, med={np.median(r_ns_filt):.0f}")
        print(f"    CALLS%   : moy={np.mean(r_cr)*100:.1f}%, med={np.median(r_cr)*100:.1f}%")
        total_r = sum(r_types.values())
        print(f"    Types    : " + ", ".join(f"{t}={v/total_r*100:.1f}%" for t, v in r_types.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_graphs", type=int, default=2000)
    parser.add_argument("--max_nodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="poc_dataset/synthetic_v2.jsonl")
    parser.add_argument("--compare_with", type=str, default=None)
    parser.add_argument("--require_call", action="store_true",
                        help="Force chaque graphe à avoir au moins 1 CALL (réduit le nombre de graphes)")
    args = parser.parse_args()

    output_path = Path(args.output)
    generate_dataset(args.n_graphs, output_path, seed=args.seed,
                     max_nodes=args.max_nodes, require_call=args.require_call)
    if args.compare_with:
        compare_with_real(output_path, Path(args.compare_with))