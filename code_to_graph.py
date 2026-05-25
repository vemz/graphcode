import ast
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset

class CodeGraphVisitor(ast.NodeVisitor):

    def __init__(self, source_code: str, file_path: str = "module"):
        self.source = source_code
        self.lines = source_code.splitlines()
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []

        self.scope_stack: List[str] = []

        self.module_id = self._make_id("module", file_path)
        self.nodes[self.module_id] = {
            "id": self.module_id,
            "name": Path(file_path).stem,
            "type": "MODULE",
            "code": "", 
            "lineno": 0,
        }
        self.scope_stack.append(self.module_id)

    @staticmethod
    def _make_id(kind: str, name: str) -> str:
        h = hashlib.md5(name.encode()).hexdigest()[:6]
        safe = name.replace(".", "_").replace("/", "_")[:40]
        return f"{kind}_{safe}_{h}"

    def _get_source(self, node) -> str:
        try:
            return ast.get_source_segment(self.source, node) or ""
        except Exception:
            return ""

    def _current_scope(self) -> str:
        return self.scope_stack[-1]

    def _add_edge(self, src: str, dst: str, etype: str):
        self.edges.append({"outNodeID": src, "inNodeID": dst, "type": etype})

    def visit_ClassDef(self, node):
        qualname = self._qualified_name(node.name)
        nid = self._make_id("class", qualname)
        self.nodes[nid] = {
            "id": nid,
            "name": node.name,
            "qualname": qualname,
            "type": "CLASS",
            "code": self._get_source(node),
            "lineno": node.lineno,
        }
        self._add_edge(self._current_scope(), nid, "CONTAINS")
        self.scope_stack.append(nid)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node):
        qualname = self._qualified_name(node.name)
        nid = self._make_id("func", qualname)
        self.nodes[nid] = {
            "id": nid,
            "name": node.name,
            "qualname": qualname,
            "type": "FUNCTION",
            "code": self._get_source(node),
            "lineno": node.lineno,
        }
        self._add_edge(self._current_scope(), nid, "CONTAINS")
        self.scope_stack.append(nid)

        for child in ast.iter_child_nodes(node):
            self.visit(child)
        self.scope_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        scope = self._current_scope()
        if scope != self.module_id:
            target = self._resolve_call_target(node)
            if target:
                self._add_edge(scope, target, "CALLS")
        self.generic_visit(node)

    def visit_Import(self, node):
        scope = self._current_scope()
        for alias in node.names:
            imp_id = self._make_id("import", alias.name)
            if imp_id not in self.nodes:
                self.nodes[imp_id] = {
                    "id": imp_id,
                    "name": alias.name,
                    "type": "IMPORT",
                    "code": f"import {alias.name}",
                    "lineno": node.lineno,
                }
            self._add_edge(scope, imp_id, "IMPORTS")

    def visit_ImportFrom(self, node):
        if not node.module:
            return
        scope = self._current_scope()
        imp_id = self._make_id("import", node.module)
        if imp_id not in self.nodes:
            self.nodes[imp_id] = {
                "id": imp_id,
                "name": node.module,
                "type": "IMPORT",
                "code": f"from {node.module} import ...",
                "lineno": node.lineno,
            }
        self._add_edge(scope, imp_id, "IMPORTS")

    def _qualified_name(self, name: str) -> str:
        parts = []
        for sid in self.scope_stack:
            n = self.nodes[sid]
            if n["type"] in ("CLASS", "FUNCTION"):
                parts.append(n["name"])
        parts.append(name)
        return ".".join(parts)

    def _resolve_call_target(self, node) -> Optional[str]:
        if isinstance(node.func, ast.Name):
            for nid, ndata in self.nodes.items():
                if ndata["type"] == "FUNCTION" and ndata["name"] == node.func.id:
                    return nid
            return None
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                for sid in reversed(self.scope_stack):
                    if self.nodes[sid]["type"] == "CLASS":
                        class_name = self.nodes[sid]["name"]
                        target_qual = f"{class_name}.{node.func.attr}"
                        for nid, ndata in self.nodes.items():
                            if ndata["type"] == "FUNCTION" and ndata.get("qualname") == target_qual:
                                return nid
                        break
            return None
        return None

def build_graph(source: str, file_path: str = "module.py") -> Tuple[Dict, List, Dict]:
    tree = ast.parse(source)
    visitor = CodeGraphVisitor(source, file_path)
    visitor.visit(tree)

    valid_ids = set(visitor.nodes.keys())
    clean_edges = [e for e in visitor.edges if e["outNodeID"] in valid_ids and e["inNodeID"] in valid_ids]

    stats = compute_stats(visitor.nodes, clean_edges)
    return visitor.nodes, clean_edges, stats


def compute_stats(nodes: Dict, edges: List) -> Dict:
    n = len(nodes)
    m = len(edges)
    type_counts = Counter(nd["type"] for nd in nodes.values())
    edge_type_counts = Counter(e["type"] for e in edges)

    degree = Counter()
    for e in edges:
        degree[e["outNodeID"]] += 1
        degree[e["inNodeID"]] += 1
    avg_degree = sum(degree.values()) / n if n else 0
    isolated = sum(1 for nid in nodes if degree[nid] == 0)

    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": round(avg_degree, 2),
        "isolated_nodes": isolated,
        "node_types": dict(type_counts),
        "edge_types": dict(edge_type_counts),
    }


def filter_graph(stats: Dict, min_nodes: int = 3, max_nodes: int = 500) -> bool:
    """Critères pour garder un graphe."""
    if stats["n_nodes"] < min_nodes or stats["n_nodes"] > max_nodes:
        return False
    non_import_edges = sum(v for k, v in stats["edge_types"].items() if k != "IMPORTS")
    if non_import_edges < 1:
        return False
    return True


def load_streaming_dataset(dataset_name: str, n_samples: int):
    if dataset_name == "codesearchnet":
        ds = load_dataset(
            "code_search_net", "python",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        def normalize(ex):
            return {
                "content": ex.get("whole_func_string", ""),
                "path": ex.get("func_name", "unknown.py"),
                "repo_name": ex.get("repository_name", "unknown"),
            }
    else:
        ds = load_dataset(
            "codeparrot/codeparrot-clean-train",
            split="train",
            streaming=True,
        )
        def normalize(ex):
            return {
                "content": ex.get("content", ""),
                "path": ex.get("path", "unknown.py"),
                "repo_name": ex.get("repo_name", "unknown"),
            }

    def gen():
        count = 0
        for ex in ds:
            if count >= n_samples:
                break
            yield normalize(ex)
            count += 1

    return gen()


def process_dataset(n_samples: int, output_path: Path,
                    min_nodes: int = 3, max_nodes: int = 500,
                    dataset_name: str = "codeparrot"):
    examples = load_streaming_dataset(dataset_name, n_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_parse_err, n_filtered = 0, 0, 0
    seen_hashes = set() 
    global_stats = {
        "total_nodes": 0,
        "total_edges": 0,
        "node_types": Counter(),
        "edge_types": Counter(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in tqdm(examples, desc="Parsing", total=n_samples):
            code = ex.get("content", "")
            if not code or len(code) < 50:
                n_filtered += 1
                continue

            h = hashlib.md5(code.encode()).hexdigest()
            if h in seen_hashes:
                n_filtered += 1
                continue
            seen_hashes.add(h)

            file_path = ex.get("path", "unknown.py")

            try:
                nodes, edges, stats = build_graph(code, file_path)
            except (SyntaxError, ValueError, RecursionError):
                n_parse_err += 1
                continue
            except Exception:
                n_parse_err += 1
                continue

            if not filter_graph(stats, min_nodes, max_nodes):
                n_filtered += 1
                continue

            record = {
                "repo": ex.get("repo_name", "unknown"),
                "file": file_path,
                "nodes": nodes,
                "edges": edges,
                "stats": stats,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1

            global_stats["total_nodes"] += stats["n_nodes"]
            global_stats["total_edges"] += stats["n_edges"]
            global_stats["node_types"].update(stats["node_types"])
            global_stats["edge_types"].update(stats["edge_types"])

    # Rapport final
    print("\n" + "=" * 60)
    print(f"EXTRACTION TERMINÉE")
    print("=" * 60)
    print(f"  Graphes sauvegardés : {n_ok}")
    print(f"  Erreurs de parsing   : {n_parse_err}")
    print(f"  Filtrés (trop petits/gros/triviaux/dup) : {n_filtered}")
    print(f"  Total examples       : {n_samples}")
    if n_ok > 0:
        print(f"\n  Nœuds totaux         : {global_stats['total_nodes']}")
        print(f"  Nœuds moy. / graphe  : {global_stats['total_nodes'] / n_ok:.1f}")
        print(f"  Arêtes totales       : {global_stats['total_edges']}")
        print(f"  Arêtes moy. / graphe : {global_stats['total_edges'] / n_ok:.1f}")
        print(f"\n  Types de nœuds : {dict(global_stats['node_types'])}")
        print(f"  Types d'arêtes : {dict(global_stats['edge_types'])}")
    print(f"\n  Output : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=400)
    parser.add_argument("--output", type=str, default="poc_dataset/graphs.jsonl")
    parser.add_argument("--min_nodes", type=int, default=3)
    parser.add_argument("--max_nodes", type=int, default=500)
    args = parser.parse_args()

    process_dataset(
        n_samples=args.n_samples,
        output_path=Path(args.output),
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )