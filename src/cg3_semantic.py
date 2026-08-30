"""CG3 plus a semantic view for few-label node classification.

Structure (CG3): a local GCN/GAT view and a global H-GCN/H-GAT view over a
coarsened hierarchy, trained with masked cross-entropy, a generative edge loss
and a contrastive loss between the two views.
Semantics: node text -> LLM descriptor -> sentence embedding -> MLP, giving a
semantic embedding and its own class logits (`src/semantic.py`).
Fusion: HSIC between the structural and semantic embeddings gates the head —
low HSIC concatenates the two embeddings and classifies them jointly, high
HSIC classifies each view and mixes the logits with entropy attention.

Usage:
  python src/cg3_semantic.py --help
  python src/cg3_semantic.py --dataset cora --budget 20 --seeds 0,1,2
  python src/cg3_semantic.py --dataset cora --semantic --texts data/cora_texts.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so this file runs as a script

from evaluation.labels import apply_label_strategy, format_budget, set_seed  # noqa: E402
from evaluation.metrics import compute_metrics  # noqa: E402
from src.hgcn import HGAT, HGCN  # noqa: E402
from src.layers import MLP, GraphAttention, GraphConvolution, identity  # noqa: E402
from src.losses import hsic_loss, masked_accuracy, masked_softmax_cross_entropy, structural_contrastive_loss  # noqa: E402
from src.preprocess import CG3Inputs, Hierarchy, build_hierarchy, build_inputs  # noqa: E402

log = logging.getLogger("cg3_semantic")

PLANETOID = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
# TAG releases used AS the graph, where the text cannot be aligned to Planetoid
TAG_NATIVE = {"citeseer_tag": "citeseer"}
# Published class names (label_texts of the TAG releases; Chen et al. 2024) —
# used in the TAPE-style prompt and by the label-leak stripper.
PLANETOID_CLASSES = {
    "cora": ["Case Based", "Genetic Algorithms", "Neural Networks", "Probabilistic Methods",
             "Reinforcement Learning", "Rule Learning", "Theory"],
    "citeseer": ["Agents", "Artificial Intelligence", "Databases", "Information Retrieval",
                 "Machine Learning", "Human-Computer Interaction"],
    "pubmed": ["Diabetes Mellitus Experimental", "Diabetes Mellitus Type 1", "Diabetes Mellitus Type 2"],
    # order = the bundle's label_texts (class index 0..5), NOT Planetoid's numbering
    "citeseer_tag": ["Agents", "Machine Learning", "Information Retrieval", "Databases",
                     "Human-Computer Interaction", "Artificial Intelligence"],
}
LOCAL_WEIGHT, GLOBAL_WEIGHT = 0.6, 0.4  # CG3's fixed mix of the local and global views
GENERATIVE_WEIGHT = 0.4  # CG3's weight on the edge-reconstruction loss


# ---------------------------------------------------------------------------
# view fusion
# ---------------------------------------------------------------------------

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def entropy_attention(logits_structural: torch.Tensor, logits_semantic: torch.Tensor):
    """Mix the two views' logits per node; the more confident view (lower
    entropy) gets the larger weight. No parameters."""
    entropy_structural = compute_entropy(logits_structural)
    entropy_semantic = compute_entropy(logits_semantic)
    attention = F.softmax(torch.stack([-entropy_structural, -entropy_semantic], dim=-1), dim=-1)
    alpha_structural, alpha_semantic = attention[:, 0:1], attention[:, 1:2]
    fused = alpha_structural * logits_structural + alpha_semantic * logits_semantic
    return fused, alpha_structural, alpha_semantic, entropy_structural, entropy_semantic


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class CG3SemanticModel(nn.Module):
    """CG3 with a semantic view.

    Structural branch (CG3): two local GCN/GAT layers and the H-GCN/H-GAT
    global view, each L2-normalised and mixed 0.6 / 0.4 into `z_structural`.
    Semantic branch: `semantic_channel` maps node texts to `z_semantic` and to
    its own class logits. Without a semantic channel the branch is all zeros
    and the model falls back to the attention head.
    Loss: masked CE + 0.4 · generative edge loss + structural contrastive loss
    + hsic_weight · HSIC + CG3's explicit L2 on the structural layers.
    """

    def __init__(self, *, input_dim: int, num_classes: int, hidden: int, local_model: str,
                 global_model: nn.Module, dropout: float, weight_decay: float,
                 temperature: float = 0.5, hp1: float = 0.9,
                 semantic_channel: nn.Module | None = None, semantic_dim: int = 128,
                 hsic_threshold: float = 0.1, hsic_sigma: float = 1.0, hsic_weight: float = 0.1,
                 hsic_max_samples: int = 1024):
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = float(weight_decay)
        self.temperature = float(temperature)
        self.hp1 = float(hp1)
        self.semantic_channel = semantic_channel
        self.semantic_dim = int(semantic_dim)
        self.hsic_threshold = float(hsic_threshold)
        self.hsic_sigma = float(hsic_sigma)
        self.hsic_weight = float(hsic_weight)
        self.hsic_max_samples = int(hsic_max_samples)

        if local_model == "gat":
            Layer, hidden_act, output_dropout = GraphAttention, F.elu, dropout
        elif local_model == "gcn":
            Layer, hidden_act, output_dropout = GraphConvolution, F.relu, 0.0
        else:
            raise ValueError(f"Unknown local_model: {local_model}")
        self.local_layers = nn.ModuleList([
            Layer(input_dim, hidden, act=hidden_act, bias=True, sparse_inputs=True, dropout=dropout),
            Layer(hidden, num_classes, act=identity, bias=True, sparse_inputs=False, dropout=output_dropout),
        ])
        self.global_model = global_model
        self.edge_decoder = MLP(2 * num_classes, 1, act=identity, bias=True)

        self.classifier_struct = nn.Linear(num_classes, num_classes)
        # The semantic-side heads exist only when the view does: a structural-only
        # run must be exactly CG3, with no extra trainable parameters (a bias-only
        # branch blended into the outputs acts as a learned class prior).
        if semantic_channel is not None:
            self.classifier_semantic = nn.Linear(semantic_dim, num_classes)
            self.classifier_fused = nn.Linear(num_classes + semantic_dim, num_classes)
        else:
            self.classifier_semantic = None
            self.classifier_fused = None

        self.state: dict = {}  # tensors of the last forward pass, for inspection

    def forward(self, inputs: CG3Inputs, labels: torch.Tensor, mask: torch.Tensor,
                tags: list[str] | None = None):
        n = inputs.features.size(0)

        # structural views
        h = self.local_layers[0](inputs.features, inputs.support)
        h = self.local_layers[1](h, inputs.support)
        z_local = F.normalize(h, p=2, dim=1)
        z_global = F.normalize(self.global_model(inputs.features), p=2, dim=1)
        z_structural = F.normalize(LOCAL_WEIGHT * z_local + GLOBAL_WEIGHT * z_global, p=2, dim=1)

        # semantic view (channels that carry their own embeddings need no texts)
        semantic_available = self.semantic_channel is not None and (
            not getattr(self.semantic_channel, "requires_texts", True)
            or (tags is not None and len(tags) == n)
        )
        logits_structural = self.classifier_struct(z_structural)
        if semantic_available:
            descriptors, x_semantic, h_semantic, logits_semantic = self.semantic_channel(tags)
            z_semantic = F.normalize(h_semantic.to(z_structural.device), p=2, dim=1)
            logits_semantic = logits_semantic.to(z_structural.device)
            loss_hsic = hsic_loss(z_structural, z_semantic, sigma=self.hsic_sigma, max_samples=self.hsic_max_samples)

            # HSIC gate and fusion
            low_hsic = loss_hsic.item() < self.hsic_threshold
            if low_hsic:
                outputs = self.classifier_fused(torch.cat([z_structural, z_semantic], dim=-1))
                alpha_structural = alpha_semantic = entropy_structural = entropy_semantic = None
            else:
                outputs, alpha_structural, alpha_semantic, entropy_structural, entropy_semantic = entropy_attention(
                    logits_structural, logits_semantic
                )
        else:
            # structural-only control: plain CG3 — no zero-vector branch, no fusion
            descriptors = x_semantic = z_semantic = logits_semantic = None
            loss_hsic = z_structural.new_zeros(())
            low_hsic = False
            outputs = logits_structural
            alpha_structural = alpha_semantic = entropy_structural = entropy_semantic = None

        # losses
        loss_ce = masked_softmax_cross_entropy(outputs, labels, mask)
        loss_gen = self._edge_loss(z_local, z_global, inputs.edge_pos)
        loss_contrastive = structural_contrastive_loss(
            z_local, z_global, inputs.train_idx, inputs.mat01_intra, inputs.mat01_inter,
            temperature=self.temperature, hp1=self.hp1,
        )
        loss = loss_ce + GENERATIVE_WEIGHT * loss_gen + loss_contrastive + self.hsic_weight * loss_hsic + self.l2()

        self.state = {
            "z_structural": z_structural.detach(),
            "z_semantic": None if z_semantic is None else z_semantic.detach(),
            "logits_structural": logits_structural.detach(),
            "logits_semantic": None if logits_semantic is None else logits_semantic.detach(),
            "alpha_structural": None if alpha_structural is None else alpha_structural.detach(),
            "alpha_semantic": None if alpha_semantic is None else alpha_semantic.detach(),
            "entropy_structural": None if entropy_structural is None else entropy_structural.detach(),
            "entropy_semantic": None if entropy_semantic is None else entropy_semantic.detach(),
            "descriptors": descriptors,
            "semantic_embeddings": None if x_semantic is None else x_semantic.detach(),
            "hsic": loss_hsic.detach(),
            "fused_by_concat": low_hsic,
        }
        terms = {
            "loss_total": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_gen": loss_gen.item(),
            "loss_contrastive": loss_contrastive.item(),
            "loss_hsic": loss_hsic.item(),
            "train_acc": masked_accuracy(outputs, labels, mask).item(),
            "fused_by_concat": float(low_hsic),
        }
        if alpha_semantic is not None:
            # entropy attention weights confidence, not correctness — keep the
            # mix inspectable so a mis-calibrated branch is visible in MLflow
            terms["alpha_semantic_mean"] = alpha_semantic.mean().item()
            terms["entropy_structural_mean"] = entropy_structural.mean().item()
            terms["entropy_semantic_mean"] = entropy_semantic.mean().item()
        return outputs, loss, terms

    def _edge_loss(self, z_local: torch.Tensor, z_global: torch.Tensor, edge_pos: torch.Tensor) -> torch.Tensor:
        """CG3's generative term: the edge decoder must recognise every edge
        (i, j) from the local view of i and the global view of j, and the
        reverse."""
        i, j = edge_pos[:, 0], edge_pos[:, 1]

        def nll(a, b):
            return -torch.log(torch.sigmoid(self.edge_decoder(torch.cat([a, b], dim=1))).clamp(min=1e-8)).mean()

        return nll(z_local[i], z_global[j]) + nll(z_global[i], z_local[j])

    def l2(self) -> torch.Tensor:
        """CG3's explicit weight decay: local layers, edge decoder, global view."""
        params = [p for layer in self.local_layers for p in layer.vars.values()] + list(self.edge_decoder.vars.values())
        return self.weight_decay * 0.5 * sum((p ** 2).sum() for p in params) + self.global_model.l2()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_planetoid(name: str, root: str):
    """Cora / CiteSeer / PubMed with the standard public split. Downloads into
    `root` on first use."""
    return Planetoid(root=root, name=PLANETOID[name.lower()])[0]


def load_tag_native(name: str, root: str = "datasets"):
    """The TAG release AS the graph (Chen et al. 2024, arXiv:2307.03393), for
    datasets whose text release cannot be aligned to Planetoid.

    citeseer_tag: 3,186 nodes / 4,225 undirected edges — a subset of
    Planetoid's 3,327 nodes, in a different order. Numbers on it are NOT
    comparable to published Planetoid-CiteSeer results; only within-experiment
    arm comparisons are valid, and the paper must say so.

    Ships with a Planetoid-style split (20/class train, 500 val, rest test)
    from the data-v1 bundle (`split_source: 'TAG release (native mode)'`).
    Features are the release's non-negative bag-of-words — safe for CG3's
    row-sum normalization and not an LLM embedding, so the semantic view is
    not leaked into X.
    """
    from torch_geometric.data import Data

    ds = TAG_NATIVE[name]
    base = Path("datasets" if root == "data" else root) / "tag" / ds
    if not (base / f"{ds}.npz").exists():
        raise SystemExit(f"{base / f'{ds}.npz'} not found — run scripts/download_data.py first")
    z = np.load(base / f"{ds}.npz", allow_pickle=True)
    und = torch.from_numpy(z["edges"].T).long()
    data = Data(
        x=torch.from_numpy(z["node_features"]).float(),
        edge_index=torch.cat([und, und.flip(0)], dim=1),  # canonical list -> both directions
        y=torch.from_numpy(z["node_labels"]).long(),
    )
    split = np.load(base / f"{ds}_planetoid_split.npz", allow_pickle=True)
    for k in ("train_mask", "val_mask", "test_mask"):
        setattr(data, k, torch.from_numpy(split[k]))
    return data


def load_texts(path: str, num_nodes: int) -> list[str]:
    """One line of raw text per node, in PyG node order."""
    texts = Path(path).read_text(encoding="utf-8").splitlines()
    if len(texts) != num_nodes:
        raise SystemExit(f"{path}: {len(texts)} lines, but the graph has {num_nodes} nodes")
    return texts


def build_semantic_channel(args, num_classes: int) -> nn.Module:
    if args.semantic_embeddings:
        from src.semantic import PrecomputedSemanticChannel

        embeddings = torch.from_numpy(np.load(args.semantic_embeddings)).float()
        return PrecomputedSemanticChannel(
            embeddings=embeddings,
            hidden_dim=args.semantic_hidden,
            semantic_dim=args.semantic_dim,
            num_classes=num_classes,
        )
    from src.semantic import GraniteDescriptorGenerator, HuggingFaceSentenceEncoder, SemanticChannel

    class_names = ([c.strip() for c in args.class_names.split(",")] if args.class_names
                   else PLANETOID_CLASSES[args.dataset])
    descriptor = None
    if args.descriptor == "tape":
        descriptor = GraniteDescriptorGenerator(
            args.descriptor_model, max_new_tokens=args.descriptor_max_tokens, class_names=class_names,
        )
    return SemanticChannel(
        descriptor_generator=descriptor,
        sentence_encoder=HuggingFaceSentenceEncoder(args.encoder_model),
        hidden_dim=args.semantic_hidden,
        semantic_dim=args.semantic_dim,
        num_classes=num_classes,
        class_names=class_names,
        strip_labels=not args.keep_label_leak,
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def build_model(args, inputs: CG3Inputs, hierarchy: Hierarchy, semantic_channel: nn.Module | None) -> CG3SemanticModel:
    Global = {"hgcn": HGCN, "hgat": HGAT}[args.global_model]
    global_model = Global(
        inputs.input_dim, inputs.num_classes, args.hidden_global, hierarchy,
        coarsen_level=args.coarsen_level, max_node_wgt=args.max_node_wgt,
        node_wgt_embed_dim=args.node_wgt_embed_dim, channel_num=args.channel_num,
        dropout=args.dropout, weight_decay=args.weight_decay,
    )
    return CG3SemanticModel(
        input_dim=inputs.input_dim, num_classes=inputs.num_classes, hidden=args.hidden_local,
        local_model=args.local_model, global_model=global_model, dropout=args.dropout,
        weight_decay=args.weight_decay, temperature=args.temperature, hp1=args.hp1,
        semantic_channel=semantic_channel, semantic_dim=args.semantic_dim,
        hsic_threshold=args.hsic_threshold, hsic_sigma=args.hsic_sigma,
        hsic_weight=args.hsic_weight, hsic_max_samples=args.hsic_max_samples,
    )


def trainable_state(model: nn.Module) -> dict:
    """The parameters worth checkpointing — not the frozen semantic backbones."""
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def predict(model: CG3SemanticModel, inputs: CG3Inputs, tags):
    model.eval()
    outputs, _, _ = model(inputs, inputs.y_train_oh, inputs.train_mask, tags)
    return outputs.argmax(dim=1), outputs


def run_seed(args, data, hierarchy: Hierarchy, tags, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    data = apply_label_strategy(data.clone(), args.label_strategy, args.budget, seed)
    inputs = build_inputs(data).to(device)
    semantic_channel = build_semantic_channel(args, inputs.num_classes) if args.semantic else None
    model = build_model(args, inputs, hierarchy, semantic_channel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)  # L2 is inside the loss

    best_val, best_epoch, best_state = -1.0, 0, trainable_state(model)
    patience_left = args.patience
    epoch_log = []
    start = time.perf_counter()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        _, loss, terms = model(inputs, inputs.y_train_oh, inputs.train_mask, tags)
        loss.backward()
        optimizer.step()

        pred, outputs = predict(model, inputs, tags)
        val_acc = (pred[inputs.val_mask] == inputs.y[inputs.val_mask]).float().mean().item()
        val_loss = F.cross_entropy(outputs[inputs.val_mask], inputs.y[inputs.val_mask]).item()
        epoch_log.append({"epoch": epoch, **terms, "val_acc": val_acc, "val_loss": val_loss})

        if val_acc > best_val:
            best_val, best_epoch, best_state = val_acc, epoch, trainable_state(model)
            patience_left = args.patience
        else:
            patience_left -= 1
            if args.early_stopping and patience_left <= 0:
                break

    model.load_state_dict(best_state, strict=False)
    pred, _ = predict(model, inputs, tags)
    metrics = compute_metrics(inputs.y[inputs.test_mask].cpu().numpy(), pred[inputs.test_mask].cpu().numpy())
    return {
        "metrics": metrics,
        "epoch_log": epoch_log,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "stopped_at": epoch,
        "runtime_sec": time.perf_counter() - start,
        "state": best_state,
    }


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

class MLflowLogger:
    """Logs to a local MLflow store, or does nothing when disabled."""

    def __init__(self, enabled: bool, uri: str, experiment: str, run_name: str, params: dict):
        self.enabled = enabled
        if not enabled:
            return
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        log.info("MLflow: %s, experiment %s, run %s", uri, experiment, run_name)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            self.mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None}, step=step)

    def end(self) -> None:
        if self.enabled:
            self.mlflow.end_run()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CG3 + semantic view for few-label node classification",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--dataset", default="cora", choices=sorted(PLANETOID) + sorted(TAG_NATIVE),
                   help="Planetoid dataset, or a TAG-native graph (citeseer_tag: the 3,186-node "
                        "text release as the graph — not comparable to Planetoid-CiteSeer numbers)")
    g.add_argument("--data-root", default="data", help="where PyG downloads the datasets")
    g.add_argument("--texts", default=None, help="raw node texts for the semantic view: one line per node, PyG order")

    g = p.add_argument_group("label budget")
    g.add_argument("--label-strategy", default="per_class", choices=("per_class", "percentage"))
    g.add_argument("--budget", type=float, default=20, help="labels per class, or a fraction of all nodes for `percentage`")
    g.add_argument("--seeds", default="0", help="comma-separated seeds; one run per seed")

    g = p.add_argument_group("structure (CG3)")
    g.add_argument("--local-model", default="gcn", choices=("gcn", "gat"))
    g.add_argument("--global-model", default="hgcn", choices=("hgcn", "hgat"))
    g.add_argument("--hidden-local", type=int, default=1024)
    g.add_argument("--hidden-global", type=int, default=32)
    g.add_argument("--coarsen-level", type=int, default=4, help="hierarchy depth of the global view")
    g.add_argument("--max-node-wgt", type=int, default=50, help="max input nodes merged into one coarse node")
    g.add_argument("--channel-num", type=int, default=4, help="channels per H-GCN / H-GAT layer")
    g.add_argument("--node-wgt-embed-dim", type=int, default=5)
    g.add_argument("--temperature", type=float, default=0.5, help="contrastive loss temperature")
    g.add_argument("--hp1", type=float, default=0.9, help="contrastive loss weight")

    g = p.add_argument_group("semantic view")
    g.add_argument("--semantic", action="store_true", help="turn the semantic view on (needs --texts)")
    g.add_argument("--semantic-dim", type=int, default=128)
    g.add_argument("--semantic-hidden", type=int, default=256, help="hidden width of the semantic MLP")
    g.add_argument("--descriptor", default="none", choices=("none", "tape"),
                   help="none = encode the node's own text (primary arm; Chen et al. 2024 Obs. 3/6); "
                        "tape = encode an LLM prediction+explanation of it (He et al. 2024)")
    g.add_argument("--class-names", default=None,
                   help="comma-separated class names for the TAPE prompt and the leak stripper; "
                        "defaults to the dataset's published names")
    g.add_argument("--keep-label-leak", action="store_true",
                   help="do NOT strip label declarations from generated explanations "
                        "(leak-control arm only — an unstripped 'Answer:' line reports "
                        "label leakage as semantic gain)")
    g.add_argument("--descriptor-model", default="ibm-granite/granite-4.2-3b", help="LLM that rewrites node text into a descriptor")
    g.add_argument("--descriptor-max-tokens", type=int, default=256)
    g.add_argument("--encoder-model", default="sentence-transformers/all-MiniLM-L6-v2", help="sentence encoder")
    g.add_argument("--semantic-embeddings", default=None,
                   help=".npy of precomputed embeddings [num_nodes, dim], PyG node order; "
                        "skips the descriptor LLM and the encoder (for API-only encoders "
                        "and for freezing one view across seeds/arms)")
    g.add_argument("--hsic-threshold", type=float, default=0.1, help="below this HSIC the views are concatenated")
    g.add_argument("--hsic-sigma", type=float, default=1.0)
    g.add_argument("--hsic-weight", type=float, default=0.1)
    g.add_argument("--hsic-max-samples", type=int, default=1024)

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--lr", type=float, default=0.01)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--dropout", type=float, default=0.6)
    g.add_argument("--early-stopping", action="store_true", help="stop when validation accuracy stalls")
    g.add_argument("--patience", type=int, default=50)
    g.add_argument("--device", default="auto", help="auto | cpu | cuda")

    g = p.add_argument_group("output")
    g.add_argument("--output", default=None, help="directory for the best checkpoint of every seed")
    g.add_argument("--epoch-log-every", type=int, default=10, help="log per-epoch metrics to MLflow every N epochs")
    g.add_argument("--no-mlflow", action="store_true")
    g.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    g.add_argument("--mlflow-experiment", default="few-label-gnn")
    return p.parse_args()


def main(args: argparse.Namespace) -> float:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    data = (load_tag_native(args.dataset, args.data_root) if args.dataset in TAG_NATIVE
            else load_planetoid(args.dataset, args.data_root))
    tags = load_texts(args.texts, data.num_nodes) if args.texts else None
    if args.semantic and tags is None and not args.semantic_embeddings:
        raise SystemExit("--semantic needs --texts or --semantic-embeddings: "
                         "Planetoid ships bag-of-words features, not the node texts")
    if args.semantic_embeddings:
        emb_rows = np.load(args.semantic_embeddings, mmap_mode="r").shape[0]
        if emb_rows != data.num_nodes:
            raise SystemExit(f"{args.semantic_embeddings}: {emb_rows} rows, "
                             f"but the graph has {data.num_nodes} nodes")
    log.info("%s: %d nodes, %d edges, %d features, %d classes; device=%s",
             args.dataset, data.num_nodes, data.num_edges, data.num_features, int(data.y.max()) + 1, device)

    start = time.perf_counter()
    hierarchy = build_hierarchy(data, args.coarsen_level, args.max_node_wgt)
    hierarchy = Hierarchy(*[[t.to(device) for t in tensors] for tensors in
                            (hierarchy.supports, hierarchy.pool, hierarchy.unpool, hierarchy.node_wgt)])
    log.info("hierarchy: %s nodes per level (%.1fs)", [s.size(0) for s in hierarchy.supports], time.perf_counter() - start)

    semantic_tag = ""
    if args.semantic:
        view = (Path(args.semantic_embeddings).stem if args.semantic_embeddings
                else f"{args.descriptor}-{args.encoder_model.split('/')[-1]}")
        semantic_tag = f"_semantic-{view}" + ("-LEAK" if args.keep_label_leak else "")
    run_name = (f"{args.dataset}_{args.local_model}-{args.global_model}"
                f"_{args.label_strategy}-{format_budget(args.budget)}{semantic_tag}")
    mlf = MLflowLogger(not args.no_mlflow, args.mlflow_uri, args.mlflow_experiment, run_name, vars(args))
    every = max(1, args.epoch_log_every)
    accs, f1s, runtimes = [], [], []
    try:
        for seed in seeds:
            result = run_seed(args, data, hierarchy, tags, seed, device)
            m = result["metrics"]
            accs.append(m["accuracy"])
            f1s.append(m["macro_f1"])
            runtimes.append(result["runtime_sec"])
            log.info("seed %d: stopped@%d best@%d val=%.4f test acc=%.4f macro-F1=%.4f (%.1fs)",
                     seed, result["stopped_at"], result["best_epoch"], result["best_val_acc"],
                     m["accuracy"], m["macro_f1"], result["runtime_sec"])

            for entry in result["epoch_log"]:
                if entry["epoch"] % every == 0 or entry["epoch"] == result["stopped_at"]:
                    mlf.log_metrics({f"seed_{seed}/{k}": v for k, v in entry.items() if k != "epoch"}, step=entry["epoch"])
            mlf.log_metrics({f"seed_{seed}/test_accuracy": m["accuracy"], f"seed_{seed}/test_macro_f1": m["macro_f1"],
                             f"seed_{seed}/best_epoch": result["best_epoch"], f"seed_{seed}/runtime_sec": result["runtime_sec"]})
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                torch.save({"args": vars(args), "state": result["state"]}, os.path.join(args.output, f"{run_name}_seed{seed}.pt"))

        n = max(len(seeds), 1)
        accs, f1s = np.array(accs), np.array(f1s)
        summary = {
            "agg/mean_accuracy": accs.mean(), "agg/std_accuracy": accs.std(), "agg/moe_accuracy": 1.96 * accs.std() / np.sqrt(n),
            "agg/mean_macro_f1": f1s.mean(), "agg/std_macro_f1": f1s.std(), "agg/moe_macro_f1": 1.96 * f1s.std() / np.sqrt(n),
            "agg/mean_runtime_sec": float(np.mean(runtimes)),
        }
        log.info("%s over %d seed(s): acc %.4f ± %.4f  macro-F1 %.4f ± %.4f  (%.1fs/seed)", run_name, n,
                 summary["agg/mean_accuracy"], summary["agg/std_accuracy"],
                 summary["agg/mean_macro_f1"], summary["agg/std_macro_f1"], summary["agg/mean_runtime_sec"])
        mlf.log_metrics(summary)
        return float(summary["agg/mean_accuracy"])
    finally:
        mlf.end()


if __name__ == "__main__":
    main(parse_args())
