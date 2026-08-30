# CG3 + a Semantic View for Few-Label Node Classification

### Overview
Node classification when only a handful of labels per class are available. The model is **CG3** (Wan et al., AAAI 2021) — a local GCN/GAT view and a global hierarchical H-GCN/H-GAT view, trained with a contrastive loss between the views and a generative edge loss — plus a **semantic view** built from each node's text (LLM descriptor → sentence embedding → MLP). **HSIC** between the structural and semantic embeddings gates how the two are fused: nearly independent views are concatenated and classified jointly; dependent views are classified separately and their logits mixed by **entropy attention**, which weights the more confident view higher.

### Installation
1. Clone this repository:
```
git clone https://github.com/Tinnifo/few-label-node-classification-gnn.git
cd few-label-node-classification-gnn
```
2. Install dependencies (Python ≥ 3.12):
```
pip install -r requirements.txt
```
or, with uv, `uv sync`.

### Datasets
Cora, CiteSeer and PubMed come from PyTorch Geometric's `Planetoid` loader with the standard public split. They download into `data/` on the first run — nothing to fetch by hand.

The semantic view needs each node's raw text, which Planetoid does not ship (its features are bag-of-words). Provide it with `--texts FILE`: one line per node, in PyG node order.

### Usage
Every option, with its default:
```
python src/cg3_semantic.py --help
```
CG3 on Cora with 20 labels per class, five seeds:
```
python src/cg3_semantic.py --dataset cora --budget 20 --seeds 0,1,2,3,4 --early-stopping
```
A 5 % label budget on PubMed:
```
python src/cg3_semantic.py --dataset pubmed --label-strategy percentage --budget 0.05
```
With the semantic view:
```
python src/cg3_semantic.py --dataset cora --semantic --texts data/cora_texts.txt
```
Runs log to a local MLflow store (`sqlite:///mlflow.db`, experiment `few-label-gnn`); browse it with `mlflow ui --backend-store-uri sqlite:///mlflow.db`, or pass `--no-mlflow`. `--output DIR` saves the best checkpoint of every seed. On SLURM, edit the parameters at the top of `sh/run_cg3_semantic.sh` and `sbatch` it.

### Repository layout
```
src/cg3_semantic.py    the model and its training script — start here
src/layers.py          GCN / GAT / MLP layers (local view), H-GCN / H-GAT layers (global view)
src/hgcn.py            H-GCN / H-GAT: the global view over the coarsened hierarchy
src/coarsening.py      MILE hybrid matching that builds the hierarchy
src/preprocess.py      PyG Data -> CG3 inputs: features, support, hierarchy, label matrices
src/losses.py          structural contrastive loss, HSIC, masked cross-entropy / accuracy
src/semantic.py        the semantic view: descriptor LLM + sentence encoder + MLP
evaluation/labels.py   few-label splits: `per_class` (k labels per class) or `percentage`
evaluation/metrics.py  accuracy and macro-F1
sh/                    SLURM launchers
```

### Citation
The structural model is CG3; if you use this code, please cite:
```
@inproceedings{wan2021contrastive,
  title={Contrastive and Generative Graph Convolutional Networks for Graph-based Semi-Supervised Learning},
  author={Wan, Sheng and Pan, Shirui and Yang, Jian and Gong, Chen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={11},
  pages={10049--10057},
  year={2021}
}
```
