import ast
from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class CallGraphVisitor(ast.NodeVisitor):
    def __init__(self, source_code: str):
        self.source_code = source_code
        self.nodes: Dict[str, dict] = {}
        self.edges: List[dict] = []
        self.current_function = None

    def _get_node_id(self, name: str) -> str:
        return f"func_{name.replace('.', '_')}"

    def visit_FunctionDef(self, node):
        node_id = self._get_node_id(node.name)
        source_segment = ast.get_source_segment(self.source_code, node) or ""
        self.nodes[node_id] = {
            "id": node_id, "name": node.name, "type": "FunctionDef",
            "code": source_segment.strip(), "lineno": node.lineno
        }
        prev = self.current_function
        self.current_function = node_id
        self.generic_visit(node)
        self.current_function = prev

    def visit_ClassDef(self, node):
        for body_node in node.body:
            if isinstance(body_node, ast.FunctionDef):
                body_node.name = f"{node.name}.{body_node.name}"
        self.generic_visit(node)

    def visit_Call(self, node):
        if self.current_function:
            if isinstance(node.func, ast.Name):
                called_id = self._get_node_id(node.func.id)
                self.edges.append({"outNodeID": self.current_function, "inNodeID": called_id, "type": "CALLS"})
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id != "self":
                    module_call = self._get_node_id(f"{node.func.value.id}_{node.func.attr}")
                    self.edges.append({"outNodeID": self.current_function, "inNodeID": module_call, "type": "CALLS"})
        self.generic_visit(node)

def build_graph(code_source: str):
    tree = ast.parse(code_source)
    visitor = CallGraphVisitor(code_source)
    visitor.visit(tree)
    return visitor.nodes, visitor.edges

def map_tokens_to_nodes(code_source: str, nodes: Dict, tokenizer):
    tokens = tokenizer(code_source, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens["input_ids"][0]
    offsets = tokens["offset_mapping"][0]

    node_positions = {}
    for node_id, node_data in nodes.items():
        snippet = node_data.get("code", "")
        if snippet:
            start = code_source.find(snippet)
            if start != -1:
                node_positions[node_id] = (start, start + len(snippet))

    token_to_node = ["GLOBAL"] * len(input_ids)
    for i, (ts, te) in enumerate(offsets):
        if ts == te == 0: continue
        for nid, (ns, ne) in node_positions.items():
            if ns <= ts < ne or ns < te <= ne:
                token_to_node[i] = nid
                break
    return input_ids, token_to_node

def get_node_initial_features(nodes: Dict, tokenizer, model, embed_dim=128):
    features = []

    word_embeddings = model.get_input_embeddings()

    for node_id, node_data in nodes.items():
        code = node_data.get("code", node_data.get("name", "unknown"))
        inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)

        with torch.no_grad():
            token_embs = word_embeddings(inputs["input_ids"])
            node_emb = token_embs.mean(dim=1).squeeze(0)

        features.append(node_emb)

    features = torch.stack(features)

    return features