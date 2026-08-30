# Experiment scaffold — semantic-view encoder ablation

Maps the `Simple` branch (`e8454b2`, the consolidated single-entrypoint layout)
onto the experiment we want to run, names the gaps that block it, and pins the
reference for every design choice.

## 1. The pipeline as it stands

One entrypoint: `src/cg3_semantic.py`. The semantic view is
**node text → LLM descriptor (Granite) → sentence embedding → MLP**, fused
with the structural CG3 embedding by an **HSIC gate**: below
`--hsic-threshold` the views are concatenated into a joint classifier; above
it, each view is classified separately and the logits are mixed per node by
**entropy attention** (more confident view weighted higher). HSIC also enters
the loss with `--hsic-weight`.

| component | file | notes |
| --- | --- | --- |
| entrypoint: CLI, train loop, MLflow, seeds | `src/cg3_semantic.py` | `parse_args` groups: data / label budget / structure / semantic / training / output |
| CG3 model + fusion | `src/cg3_semantic.py` (`CG3SemanticModel`, `entropy_attention`) | local GCN/GAT + global H-GCN/H-GAT, contrastive + generative + CE + HSIC terms |
| semantic channel | `src/semantic.py` (`GraniteDescriptorGenerator`, `HuggingFaceSentenceEncoder`, `SemanticChannel`) | descriptor LLM: `ibm-granite/granite-4.2-3b`; encoder: `all-MiniLM-L6-v2` (both CLI-swappable) |
| CG3 losses, HSIC | `src/losses.py` | real RBF-kernel `hsic_loss` (subsampled at `--hsic-max-samples`) |
| global-view hierarchy | `src/coarsening.py`, `src/hgcn.py`, `src/layers.py`, `src/preprocess.py` | MILE-style coarsening; `Hierarchy`, `CG3Inputs` |
| data | PyG `Planetoid` (auto-download) + `--texts` file | texts: one line per node, **PyG node order** |
| label budgets | `evaluation/labels.py` (`set_few_label_mask`, `set_budget_percent`) | see gap 4 |
| metrics | `evaluation/metrics.py` | acc + macro-F1 |
| cluster launch | `sh/run_cg3_semantic.sh` | SLURM; `SEMANTIC=""` toggles the view |
| data fetch | `scripts/download_data.py` (this branch) | release `data-v1`, sha256-verified |
| texts adapter | `scripts/make_texts.py` (this branch) | TAG `.npz` → `--texts` file, PyG order |

## 2. The experiment

**Question.** Does the semantic view help CG3 at few labels, and how much of
the answer is encoder (and descriptor) capacity?

**Arms** (per dataset × budget, ≥10 seeds) — flags as implemented on this branch:

| arm | flags | view |
| --- | --- | --- |
| A0 control | *(no `--semantic`)* | none — bit-equal to faithful CG3 (fusion bypassed, no semantic parameters) |
| A1 primary | `--semantic --texts datasets/tag/<ds>/<ds>_texts.txt` | own text → MiniLM-L6-v2 (384d); `--descriptor none` is the default |
| A2 | A1 + `--encoder-model intfloat/e5-large-v2` | stronger open encoder (1024d; `query:` prefix applied automatically) |
| A2′ | `--semantic --semantic-embeddings datasets/tag/<ds>/<ds>_sem_gpt3l.npy` | text-embedding-3-large (3072d), precomputed |
| A3 | A1 + `--descriptor tape` | TAPE-style LLM prediction+explanation, label-leak stripped |
| A4 leak control | A3 + `--keep-label-leak` | unstripped explanations — measures how much "gain" is leakage |

Grounds: encoder capacity was the dominant axis in the July probe (gpt3l
+0.067 vs sbert +0.017 vs explanation view +0.005) and in Chen et al.
(Obs. 3/6: frozen sentence embeddings + GNN beat fine-tuned PLMs; e5-large
top). A3 is the descriptor-ablation analogue of TAPE's "explanation vs
original text" comparison, run at our label budgets.

**Fixed, not ablated** — with the citation that licenses fixing it:

- **Descriptor LLM = Granite-4.2-3b.** TAPE Table 4 (GPT-3.5 ↔ Llama2-13b)
  shows explanation-style features are robust to the LLM backbone; sweeping it
  buys nothing. If a reviewer asks, that table is the answer.
- **Descriptor prompt**: plain rewrite, no chain-of-thought (Chen et al.
  Obs. 13: unreliable), no neighbor summaries (Obs. 15–16: homophily-gated,
  hurts PubMed).
- **Encoder fine-tuning: none** (Chen et al.: ~20 pts below frozen at low
  label).
- **Stopping rule**: pass `--early-stopping --patience 200` explicitly and
  identically on every arm. Default patience is 50 — the July convergence
  measurement (best epochs 21–167 across seeds on pubmed) says 50 truncates;
  and the stopping rule must never differ between arms (measured confound
  ≈600× the seed noise floor).

**Leakage rule.** Any generated descriptor must be checked for label
declarations before encoding (the July measurement: an unstripped
"Answer: …" line accounts for most of an apparent explanation-view gain;
prefix-matching alone catches only ~60–70% — strip structurally).

## 3. Closed on this branch

1. **A0 control bypass** — with the view off, `outputs = logits_structural`;
   the semantic heads are not even constructed, so the control has zero extra
   parameters (previously a trainable bias branch was entropy-mixed into every
   output, i.e. a learned class prior CG3 does not have).
2. **Texts adapter** — `scripts/make_texts.py` writes one-line-per-node files
   in PyG order from the TAG bundles (cora/pubmed are Planetoid-aligned; edge
   sets match PyG exactly: 5,278 / 44,324 undirected).
3. **Precomputed-view bypass** — `--semantic-embeddings <npy>` skips
   descriptor+encoder (row count validated against the graph).
4. **Descriptor/embedding caching** — the frozen LLM+encoder pass now runs
   once per run, not once per epoch and validation call (it previously
   re-sampled descriptors every forward: hours of inference per run and a
   semantic view that changed under the model). Decoding is greedy, so the
   explanation features are reproducible.
5. **TAPE-format prompt + structural leak stripper** — `--descriptor tape`
   uses the closed-category prediction+explanation prompt;
   `strip_label_declarations` removes leading declaration blocks, bare class
   lists, and per-class headers (prefix-only stripping catches ~60–70%);
   `--keep-label-leak` preserves them for the A4 control.
6. **Budget guard** — an unsatisfiable `--budget` now refuses loudly instead
   of silently capping at the public pool's 20/class.
7. **Fusion diagnostics** — `alpha_semantic_mean` and per-view entropies land
   in MLflow next to `fused_by_concat` and `loss_hsic`.

## 3b. Still open

- **CiteSeer**: the TAG release is 3,186 nodes vs PyG's 3,327 —
  `make_texts.py` refuses it for the Planetoid path. Decide: skip citeseer in
  semantic arms, or add a TAG-native graph path.
- **Acceptance run**: A0 must reproduce the faithful-port CG3 numbers
  seed-for-seed on the cluster before any arm comparison is trusted.
- **HSIC gate behavior**: the gate flips discretely at `--hsic-threshold`
  (HSIC is not scale-invariant — Gretton et al. 2005 — so the threshold is
  coupled to `--hsic-sigma` and embedding dim; both views are L2-normalized
  first, which helps). Watch `fused_by_concat` across epochs. Entropy
  attention weights **confidence, not correctness** (July probe: −26 pts when
  confidence anti-correlates with accuracy; per-branch temperature scaling on
  validation recovers it) — the new diagnostics make this checkable.

## 4. Data

`python scripts/download_data.py` fetches release
[`data-v1`](https://github.com/Tinnifo/few-label-node-classification-gnn/releases/tag/data-v1)
(sha256-pinned): classic Planetoid `ind.*` (byte-identical to
`kimiyoung/planetoid`; PyG's loader fetches the same files itself — the asset
is frozen provenance) and the TAG bundles with `node_texts` plus four
precomputed views (`sbert`, `gpt3l`, `tape_full`, `tape_stripped`). Then
`python scripts/make_texts.py cora pubmed` writes the `--texts` files.

## 5. References

| choice | cite |
| --- | --- |
| Planetoid data + split | Yang, Cohen & Salakhutdinov, ICML 2016. arXiv:1603.08861 |
| PyG loader | Fey & Lenssen, *Fast Graph Representation Learning with PyTorch Geometric*, 2019. arXiv:1903.02428 |
| raw node text (TAG) | Chen et al., SIGKDD Explorations 25, 2024. arXiv:2307.03393 (also the encoder-choice evidence: Obs. 3/6, low-label Table 5) |
| base method (CG3) | Wan et al., AAAI 2021. arXiv:2009.07111 |
| global view (H-GCN) | Hu et al., IJCAI 2019. arXiv:1902.06667 |
| local view (GCN / GAT) | Kipf & Welling, ICLR 2017. arXiv:1609.02907 / Veličković et al., ICLR 2018. arXiv:1710.10903 |
| descriptor-as-feature idea | He et al. (TAPE), ICLR 2024. arXiv:2305.19523 — also the LLM-robustness table and the post-cutoff (tape-arxiv23) contamination precedent |
| descriptor LLM | IBM Granite (cite the model card / technical report for the pinned checkpoint) |
| sentence encoder (default) | Reimers & Gurevych (Sentence-BERT), EMNLP 2019. arXiv:1908.10084; MiniLM: Wang et al., NeurIPS 2020. arXiv:2002.10957 |
| sentence encoder (A2) | Wang et al. (E5), 2022. arXiv:2212.03533 |
| encoder A2′ | OpenAI text-embedding-3-large, Jan 2024 release note (no paper) |
| HSIC | Gretton et al., ALT 2005 — *Measuring Statistical Dependence with Hilbert-Schmidt Norms* |
| entropy-weighted fusion | Han et al., Findings of ACL 2026 — entropy-guided uncertainty fusion (mechanism precedent) |
| when LLMs help (scope) | Wu et al. (LLMNodeBed), ICML 2025. arXiv:2502.00829 |
| homophily descriptors (h_adj, LI) | Platonov et al., NeurIPS 2023. arXiv:2209.06177 |
