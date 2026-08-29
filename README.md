DATA
- AG  (stub: `from data import load_ag`)
- TAG (stub: `from data import load_tag`)

MODELS  (`from src.model import GCN, HGCN, HGAT`)
- GCN
- HGCN / HGAT (CG3 global view)

METHOD  (`from src.method import SemanticGNNModel`)
- adapted CG3 + semantic view

UTILS
- `utils/graph.py`  original CG3
- `utils/losses.py` view losses
- `utils/transformers.py` semantic channel
- `utils/mlflow_logger.py`

SH  (`import sh`)
- `sh/run_cg3.sh`

main.py
- data load + label strategy
- train model
- inference
- eval
- plot / log mlflow
