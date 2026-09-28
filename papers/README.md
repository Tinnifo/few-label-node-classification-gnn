# Papers

Reading list for the project. **Links only** — this repository is public, so paper PDFs are not committed here
(that would redistribute copyrighted work). Every link goes to the arXiv abstract page where one exists, otherwise to the publisher DOI.

- **Part 1 — Papers we have discussed**: the ones the work in [`PROGRESS.md`](../PROGRESS.md) actually relies on, grouped by the question they bear on.
- **Part 2 — Full reading shelf**: all 216 papers on the project's shelf, alphabetical.

_Last updated 2026-09-27. arXiv IDs are checked against arXiv's own titles._

## Part 1 — Papers we have discussed

### When can a semantic view help? (assumption 1)

| Paper | Why it matters here | Code |
| --- | --- | --- |
| Wu et al. 2025 — [When Do LLMs Help With Node Classification? A Comprehensive Analysis](https://arxiv.org/abs/2502.00829)  | encoder gain falls with homophily (ρ = −0.872, 13 datasets); a benchmark of LLM methods for node classification | [WxxShirley/LLMNodeBed](https://github.com/WxxShirley/LLMNodeBed) |
| Li et al. 2024 — [GLBench: A Comprehensive Benchmark for Graph with Large Language Models](https://arxiv.org/abs/2407.07457) [DOI](https://doi.org/10.52202/079017-1340) | at 20/class the fused model barely beats the better single view | [NineAbyss/GLBench](https://github.com/NineAbyss/GLBench) |
| Kornblith et al. 2019 — [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)  | linear CKA, the 0-to-1 similarity score we use in Exp 5 | [google-research/google-research/tree/master/representation_similarity](https://github.com/google-research/google-research/tree/master/representation_similarity) |
| Li & Flanigan 2023 — [Task Contamination: Language Models May Not Be Few-Shot Anymore](https://arxiv.org/abs/2312.16337) [DOI](https://doi.org/10.1609/aaai.v38i16.29808) | how to audit an encoder for pre-training contamination |  |
| Platonov et al. 2023 — [Characterizing Graph Datasets for Node Classification: Homophily–Heterophily Dichotomy and Beyond](https://arxiv.org/abs/2209.06177) [DOI](https://doi.org/10.52202/075280-0025) | label informativeness: a candidate explanation for why PubMed gains despite high homophily (Exp 5) | [https://colab.research.google.com/drive/186KlV8PrWOq_woZVaRWGYC2B10vSm7S2?usp=sharing](https://colab.research.google.com/drive/186KlV8PrWOq_woZVaRWGYC2B10vSm7S2?usp=sharing) |
| Platonov et al. 2023 — [A Critical Look at the Evaluation of GNNs Under Heterophily: Are We Really Making Progress?](https://arxiv.org/abs/2302.11640)  | the standard heterophilic benchmarks are flawed; replacement datasets | [yandex-research/heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs) |
| Zhang et al. 2025 — [Leveraging Large Language Models for Effective Label-free Node Classification in Text-Attributed Graphs](https://doi.org/10.1145/3726302.3730021)  | label-free node classification with LLMs (Locle) | [HKBU-LAGAS/Locle](https://github.com/HKBU-LAGAS/Locle) |
| Li et al. 2024 — [Enhancing Graph Neural Networks with Limited Labeled Data by Actively Distilling Knowledge from Large Language Models](https://arxiv.org/abs/2407.13989) [DOI](https://doi.org/10.1109/bigdata62323.2024.10825477) | distilling LLM knowledge into GNNs under few labels |  |

### Combining the views (assumptions 2 and 4)

| Paper | Why it matters here | Code |
| --- | --- | --- |
| Wang et al. 2020 — [AM-GCN: Adaptive Multi-channel Graph Convolutional Networks](https://arxiv.org/abs/2007.02265) [DOI](https://doi.org/10.1145/3394486.3403177) | attention fusion with shared and private embeddings; the design behind our shared/private arm (Exp 4) | [zhumeiqiBUPT/AM-GCN](https://github.com/zhumeiqiBUPT/AM-GCN) |

### Honest evaluation and calibration (assumption 3)

| Paper | Why it matters here | Code |
| --- | --- | --- |
| Oliver et al. 2018 — [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170)  | validation labels count as labels when you report a label budget | [brain-research/realistic-ssl-evaluation](https://github.com/brain-research/realistic-ssl-evaluation) |
| Shchur et al. 2018 — [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868)  | variation across seeds and splits sets the noise floor | [shchur/gnn-benchmark](https://github.com/shchur/gnn-benchmark) |
| Guo et al. 2017 — [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)  | temperature scaling |  |
| Wang et al. 2021 — [Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration (CaGCN)](https://arxiv.org/abs/2109.14285)  | CaGCN — GNN calibration |  |
| Hsu et al. 2022 — [What Makes Graph Neural Networks Miscalibrated? (GATS)](https://arxiv.org/abs/2210.06391)  | GATS — GNN calibration |  |
| Dawood et al. 2023 — [Uncertainty aware training to improve deep learning model calibration for classification of cardiac MR images](https://arxiv.org/abs/2308.15141) [DOI](https://doi.org/10.1016/j.media.2023.102861) | calibration-aware training (read in depth) | [tareend/Uncertainty-Aware-Training](https://github.com/tareend/Uncertainty-Aware-Training) |

### Baselines to compare against (assumption 5)

| Paper | Why it matters here | Code |
| --- | --- | --- |
| Song & King 2026 — [Semi-supervised Instruction Tuning for Large Language Models on Text-Attributed Graphs (SIT-Graph)](https://arxiv.org/abs/2601.12807)  | few-shot TAG baseline (moves labels, not text) |  |
| Kim et al. 2026 — [Who Should Teach? Confidence-Aware Dual-Teacher Learning for Few-Shot Node Classification on Text-Attributed Graphs (CoTeach)](https://arxiv.org/abs/2608.22127)  | few-shot TAG baseline |  |
| Wei et al. 2025 — [Preference-driven Knowledge Distillation for Few-shot Node Classification (PKD)](https://arxiv.org/abs/2510.10116)  | few-shot TAG baseline |  |
| Zhao et al. 2023 — [Learning on Large-scale Text-attributed Graphs via Variational Inference (GLEM)](https://arxiv.org/abs/2210.14709)  | fine-tuned LM baseline |  |
| Duan et al. 2023 — [SimTeG: A Frustratingly Simple Approach Improves Textual Graph Learning](https://arxiv.org/abs/2308.02565)  | fine-tuned LM baseline | [vermouthdky/SimTeG](https://github.com/vermouthdky/SimTeG) |
| He et al. 2024 — [Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning](https://arxiv.org/abs/2305.19523)  | TAPE: the source of our LLM-explanation (tape) views | [XiaoxinHe/TAPE](https://github.com/XiaoxinHe/TAPE) |

### Prior work related to the LLM feedback loop (literature check, 2026-09-08)

| Paper | Why it matters here | Code |
| --- | --- | --- |
| Ji et al. 2024 — [Verbalized Graph Representation Learning: A Fully Interpretable Graph Model Based on Large Language Models Throughout the Entire Process](https://arxiv.org/abs/2410.01457)  | VGRL: the closest published work to the LLM feedback loop | [https://anonymous.4open.science/r/VGRL-7E1E](https://anonymous.4open.science/r/VGRL-7E1E) |
| Qiao et al. 2024 — [LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework](https://arxiv.org/abs/2405.13902) [DOI](https://doi.org/10.1145/3701551.3703488) | LLM consulted on the GNN's low-confidence nodes | [QiaoYRan/LOGIN](https://github.com/QiaoYRan/LOGIN) |
| Pryzant et al. 2023 — [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.494) | ProTeGi — errors → natural-language gradient | [microsoft/LMOps/tree/main/prompt_optimization](https://github.com/microsoft/LMOps/tree/main/prompt_optimization) |
| Yuksekgonul et al. 2024 — [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496)  | text as the optimised variable |  |
| Xiao et al. 2024 — [Verbalized Machine Learning: Revisiting Machine Learning with Language Models](https://arxiv.org/abs/2406.04344)  | VML — the model is text | [timxzz/VML_Examples](https://github.com/timxzz/VML_Examples) |
| Ye et al. 2022 — [PROGEN: Progressive Zero-shot Dataset Generation via In-context Feedback](https://arxiv.org/abs/2210.12329) [DOI](https://doi.org/10.18653/v1/2022.findings-emnlp.269) | small-model errors feed the LLM's next generation round | [HKUNLP/ProGen](https://github.com/HKUNLP/ProGen) |
| Wang et al. 2023 — [Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models](https://arxiv.org/abs/2310.13671)  | synthetic-data loop driven by small-model errors |  |
| Ludan et al. 2023 — [Interpretable-by-Design Text Understanding with Iteratively Generated Concept Bottleneck (TBM)](https://arxiv.org/abs/2310.19660)  | TBM — concepts grown from misclassified examples |  |
| Esfandiarpoor et al. 2024 — [Follow-up Differential Descriptions: Language Models Resolve Ambiguities for Image Classification](https://arxiv.org/abs/2311.07593)  | FuDD — descriptions for confused class pairs | [BatsResearch/fudd](https://github.com/BatsResearch/fudd) |

### Foundations we have read closely

| Paper | Why it matters here | Code |
| --- | --- | --- |
| Kipf & Welling 2017 — [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)  | GCN: the graph backbone (read in depth) | [tkipf/gcn](https://github.com/tkipf/gcn) |
| Wan et al. 2021 — [Contrastive and Generative Graph Convolutional Networks for Graph-based Semi-Supervised Learning](https://doi.org/10.1609/aaai.v35i11.17206)  | CG3: our base model (read in depth) | [LEAP-WS/CG3](https://github.com/LEAP-WS/CG3) |
| Gretton et al. 2005 — [Measuring Statistical Dependence with Hilbert-Schmidt Norms](https://doi.org/10.1007/11564089_7)  | HSIC: the dependence penalty used in Exp 1 and 4 (read in depth) |  |
| van den Oord et al. 2018 — [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)  | InfoNCE (contrastive-learning background) |  |
| Chen et al. 2020 — [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)  | SimCLR (contrastive-learning background) | [google-research/simclr](https://github.com/google-research/simclr) |
| Wang & Isola 2020 — [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242)  | alignment/uniformity (contrastive-learning background) | [SsnL/align_uniform](https://github.com/SsnL/align_uniform) |

## Part 2 — Full reading shelf

| Paper | Year | Also | Code |
| --- | --- | --- | --- |
| [A cost function for similarity-based hierarchical clustering](https://arxiv.org/abs/1510.05043) | 2016 | [DOI](https://doi.org/10.1145/2897518.2897527) |  |
| [A Critical Look at the Evaluation of GNNs Under Heterophily: Are We Really Making Progress?](https://arxiv.org/abs/2302.11640) | 2024 |  | [yandex-research/heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs) |
| [A Generalization of Transformer Networks to Graphs](https://arxiv.org/abs/2012.09699) | 2021 |  | [graphdeeplearning/graphtransformer](https://github.com/graphdeeplearning/graphtransformer) |
| [A Generalization Theory for JEPA-Based World Models](https://arxiv.org/abs/2606.27014) | 2026 |  |  |
| [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) | 2020 |  | [google-research/simclr](https://github.com/google-research/simclr) |
| [A Survey of Few-Shot Learning on Graphs: from Meta-Learning to Pre-Training and Prompt Learning](https://arxiv.org/abs/2402.01440) | 2025 |  | [smufang/fewshotgraph (paper list / survey website)](https://github.com/smufang/fewshotgraph (paper list / survey website)) |
| [A Survey of Graph Meets Large Language Model: Progress and Future Directions](https://arxiv.org/abs/2311.12399) | 2023 | [DOI](https://doi.org/10.24963/ijcai.2024/898) |  |
| [A Survey on Deep Active Learning: Recent Advances and New Frontiers](https://arxiv.org/abs/2405.00334) | 2024 | [DOI](https://doi.org/10.1109/tnnls.2024.3396463) |  |
| [Active Discriminative Network Representation Learning](https://doi.org/10.24963/ijcai.2018/296) | 2018 |  |  |
| [Active Learning for Graph Embedding](https://arxiv.org/abs/1705.05085) | 2017 |  | [vwz/AGE](https://github.com/vwz/AGE) |
| [Active Learning for Graph Neural Networks via Node Feature Propagation](https://arxiv.org/abs/1910.07567) | 2019 |  | [CrickWu/active_graph](https://github.com/CrickWu/active_graph) |
| [Active Learning for Graphs with Noisy Structures](https://arxiv.org/abs/2402.02321) | 2024 |  |  |
| [Active World Model Learning with Progress Curiosity](https://arxiv.org/abs/2007.07853) | 2020 |  |  |
| [ALINC: Active Learning for Inductive Node Classification via Graph Sampling](https://arxiv.org/abs/2606.04647) | 2026 | [DOI](https://doi.org/10.1007/978-3-032-37654-1_6) | [pasplett/alinc](https://github.com/pasplett/alinc) |
| [All in One: Multi-Task Prompting for Graph Neural Networks](https://arxiv.org/abs/2307.01504) | 2023 | [DOI](https://doi.org/10.1145/3580305.3599256) | [sheldonresearch/ProG](https://github.com/sheldonresearch/ProG) |
| [AM-GCN: Adaptive Multi-channel Graph Convolutional Networks](https://arxiv.org/abs/2007.02265) | 2020 | [DOI](https://doi.org/10.1145/3394486.3403177) | [zhumeiqiBUPT/AM-GCN](https://github.com/zhumeiqiBUPT/AM-GCN) |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | [DOI](https://doi.org/10.65215/2q58a426) | [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) |
| [Augmentation-Free Self-Supervised Learning on Graphs](https://arxiv.org/abs/2112.02472) | 2022 | [DOI](https://doi.org/10.1609/aaai.v36i7.20700) | [Namkyeong/AFGRL](https://github.com/Namkyeong/AFGRL) |
| [Augmenting Low-Resource Text Classification with Graph-Grounded Pre-training and Prompting](https://arxiv.org/abs/2305.03324) | 2023 | [DOI](https://doi.org/10.1145/3539618.3591641) | [WenZhihao666/G2P2](https://github.com/WenZhihao666/G2P2) |
| [Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496) | 2024 |  | [zou-group/textgrad](https://github.com/zou-group/textgrad) |
| [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) | 2023 | [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.494) | [microsoft/LMOps/tree/main/prompt_optimization](https://github.com/microsoft/LMOps/tree/main/prompt_optimization) |
| [BANGS: Game-Theoretic Node Selection for Graph Self-Training](https://arxiv.org/abs/2410.09348) | 2024 |  | [fangxin-wang/BANGS](https://github.com/fangxin-wang/BANGS) |
| [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) | 2021 |  | [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins) |
| [Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs](https://arxiv.org/abs/2006.11468) | 2020 |  | [GemsLab/H2GCN](https://github.com/GemsLab/H2GCN) |
| [Bridging Local Details and Global Context in Text-Attributed Graphs](https://arxiv.org/abs/2406.12608) | 2024 | [DOI](https://doi.org/10.18653/v1/2024.emnlp-main.823) | [wykk00/GraphBridge](https://github.com/wykk00/GraphBridge) |
| [Can GNN be Good Adapter for LLMs?](https://arxiv.org/abs/2402.12984) | 2024 | [DOI](https://doi.org/10.1145/3589334.3645627) | [zjunet/GraphAdapter](https://github.com/zjunet/GraphAdapter) |
| [Can Large Language Models Improve the Adversarial Robustness of Graph Neural Networks?](https://arxiv.org/abs/2408.08685) | 2024 |  | [zhongjian-zhang/LLM4RGNN](https://github.com/zhongjian-zhang/LLM4RGNN) |
| [Can LLMs Effectively Leverage Graph Structural Information through Prompts in Text-Attributed Graphs, and Why?](https://arxiv.org/abs/2309.16595) | 2023 |  | [TRAIS-Lab/LLM-Structured-Data](https://github.com/TRAIS-Lab/LLM-Structured-Data) |
| [Characterizing Graph Datasets for Node Classification: Homophily–Heterophily Dichotomy and Beyond](https://arxiv.org/abs/2209.06177) | 2024 | [DOI](https://doi.org/10.52202/075280-0025) | [https://colab.research.google.com/drive/186KlV8PrWOq_woZVaRWGYC2B10vSm7S2?usp=sharing](https://colab.research.google.com/drive/186KlV8PrWOq_woZVaRWGYC2B10vSm7S2?usp=sharing) |
| [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815) | 2024 |  | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| [Class-Balanced and Reinforced Active Learning on Graphs](https://arxiv.org/abs/2402.10074) | 2024 |  |  |
| [CO-EVOLVE: Bidirectional Co-Evolution of Graph Structure and Semantics for Heterophilous Learning](https://arxiv.org/abs/2603.19596) | 2026 |  |  |
| [CoACL: Coupled Augmentation for Contrastive Learning on Text-Attributed Graphs Under Semantic Supervision from Large Language Models](https://doi.org/10.3390/electronics15040844) | 2026 |  |  |
| [Collective Classification in Network Data](https://doi.org/10.1609/aimag.v29i3.2157) | 2008 |  |  |
| [ConGraT: Self-Supervised Contrastive Pretraining for Joint Graph and Text Embeddings](https://doi.org/10.18653/v1/2024.textgraphs-1.2) | 2024 |  | [wwbrannon/congrat](https://github.com/wwbrannon/congrat) |
| [Contrastive and Generative Graph Convolutional Networks for Graph-based Semi-Supervised Learning](https://doi.org/10.1609/aaai.v35i11.17206) | 2020 |  | [LEAP-WS/CG3](https://github.com/LEAP-WS/CG3) |
| [Contrastive Learning of Structured World Models](https://arxiv.org/abs/1911.12247) | 2020 |  | [tkipf/c-swm](https://github.com/tkipf/c-swm) |
| [Contrastive Meta-Learning for Few-shot Node Classification](https://arxiv.org/abs/2306.15154) | 2023 | [DOI](https://doi.org/10.1145/3580305.3599288) | [SongW-SW/COSMIC](https://github.com/SongW-SW/COSMIC) |
| [Contrastive Multi-View Representation Learning on Graphs](https://arxiv.org/abs/2006.05582) | 2020 |  | [kavehhassani/mvgrl](https://github.com/kavehhassani/mvgrl) |
| [Cost-aware LLM-based Online Dataset Annotation](https://arxiv.org/abs/2505.15101) | 2025 |  |  |
| [CoVar: Confidence–Variance-Guided Pseudo-Label Selection for Semi-Supervised Learning](https://arxiv.org/abs/2601.11670) | 2026 |  | [ljs11528/CoVar_Pseudo_Label_Selection](https://github.com/ljs11528/CoVar_Pseudo_Label_Selection) |
| [Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131) | 2020 |  | [CRIPAC-DIG/GRACE](https://github.com/CRIPAC-DIG/GRACE) |
| [Deep Graph Infomax](https://arxiv.org/abs/1809.10341) | 2019 | [DOI](https://doi.org/10.17863/cam.40744) | [PetarV-/DGI](https://github.com/PetarV-/DGI) |
| [Deep Semantic Graph Learning via LLM based Node Enhancement](https://arxiv.org/abs/2502.07982) | 2025 |  |  |
| [DiffusAL: Coupling Active Learning with Graph Diffusion for Label-Efficient Node Classification](https://arxiv.org/abs/2308.00146) | 2023 | [DOI](https://doi.org/10.1007/978-3-031-43412-9_5) | [lmu-dbs/diffusal](https://github.com/lmu-dbs/diffusal) |
| [Disentangle-then-Refine: LLM-Guided Decoupling and Structure-Aware Refinement for Graph Contrastive Learning](https://arxiv.org/abs/2604.14746) | 2026 |  |  |
| [Dissimilar Nodes Improve Graph Active Learning](https://arxiv.org/abs/2212.01968) | 2022 |  | [franklinnwren/DS-AGE](https://github.com/franklinnwren/DS-AGE) |
| [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234) | 2021 |  | [microsoft/Graphormer](https://github.com/microsoft/Graphormer) |
| [Do We Really Need Graph Neural Networks for Traffic Forecasting?](https://arxiv.org/abs/2301.12603) |  |  |  |
| [Domain Separation Networks](https://arxiv.org/abs/1608.06019) | 2016 |  | [tensorflow/models](https://github.com/tensorflow/models) |
| [Dreamer-CDP: Improving Reconstruction-free World Models Via Continuous Deterministic Representation Prediction](https://arxiv.org/abs/2603.07083) | 2026 |  | [fmi-basel/Dreamer-CDP](https://github.com/fmi-basel/Dreamer-CDP) |
| [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) | 2013 |  | [https://code.google.com/p/word2vec (defunct) — archived at tmikolov/word2vec](https://code.google.com/p/word2vec (defunct) — archived at https://github.com/tmikolov/word2vec) |
| [Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment](https://arxiv.org/abs/2506.07168) | 2025 |  |  |
| [Efficient Tuning and Inference for Large Language Models on Textual Graphs](https://arxiv.org/abs/2401.15569) | 2024 | [DOI](https://doi.org/10.24963/ijcai.2024/634) | [ZhuYun97/ENGINE](https://github.com/ZhuYun97/ENGINE) |
| [Emergence of Invariance and Disentanglement in Deep Representations](https://arxiv.org/abs/1706.01350) | 2018 | [DOI](https://doi.org/10.1109/ita.2018.8503149) |  |
| [Enhancing Graph Neural Networks with Limited Labeled Data by Actively Distilling Knowledge from Large Language Models](https://arxiv.org/abs/2407.13989) | 2024 | [DOI](https://doi.org/10.1109/bigdata62323.2024.10825477) |  |
| [Enhancing Semi-Supervised Multi-View Graph Convolutional Networks via Supervised Contrastive Learning and Self-Training](https://arxiv.org/abs/2512.13770) | 2025 |  | [HuaiyuanXiao/MVSupGCN](https://github.com/HuaiyuanXiao/MVSupGCN) |
| [Enhancing Spectral Graph Neural Networks with LLM-Predicted Homophily](https://arxiv.org/abs/2506.14220) | 2025 |  |  |
| [ERAlign: Energy-based Representation Alignment of GNNs and LLMs on Text-attributed Graphs](https://arxiv.org/abs/2606.10461) | 2026 |  |  |
| [Exploiting Text Semantics for Few and Zero Shot Node Classification on Text-attributed Graph](https://arxiv.org/abs/2505.08168) | 2025 | [DOI](https://doi.org/10.24963/ijcai.2025/385) |  |
| [Exploring the Potential of Large Language Models (LLMs) in Learning on Graphs](https://arxiv.org/abs/2307.03393) | 2024 |  | [CurryTang/Graph-LLM](https://github.com/CurryTang/Graph-LLM) |
| [Few-Shot Learning with Graph Neural Networks](https://arxiv.org/abs/1711.04043) | 2018 |  | [vgsatorras/few-shot-gnn](https://github.com/vgsatorras/few-shot-gnn) |
| [Few-shot Node Classification with Extremely Weak Supervision](https://arxiv.org/abs/2301.02708) | 2023 | [DOI](https://doi.org/10.1145/3539597.3570435) | [SongW-SW/X-FNC](https://github.com/SongW-SW/X-FNC) |
| [Follow-up Differential Descriptions: Language Models Resolve Ambiguities for Image Classification](https://arxiv.org/abs/2311.07593) | 2024 |  | [BatsResearch/fudd](https://github.com/BatsResearch/fudd) |
| [FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models](https://arxiv.org/abs/2311.15614) | 2023 | [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.896) | [Justherozen/FreeAL](https://github.com/Justherozen/FreeAL) |
| [From Canonical Correlation Analysis to Self-supervised Graph Neural Networks](https://arxiv.org/abs/2106.12484) | 2021 |  | [hengruizhang98/CCA-SSG](https://github.com/hengruizhang98/CCA-SSG) |
| [From Selection to Generation: A Survey of LLM-based Active Learning](https://arxiv.org/abs/2502.11767) | 2025 | [DOI](https://doi.org/10.18653/v1/2025.acl-long.708) |  |
| [GALAXY: Graph-based Active Learning at the Extreme](https://arxiv.org/abs/2202.01402) | 2022 |  | [jifanz/GALAXY](https://github.com/jifanz/GALAXY) |
| [GAugLLM: Improving Graph Contrastive Learning for Text-Attributed Graphs with Large Language Models](https://arxiv.org/abs/2406.11945) | 2024 | [DOI](https://doi.org/10.1145/3637528.3672035) | [NYUSHCS/GAugLLM](https://github.com/NYUSHCS/GAugLLM) |
| [GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs](https://arxiv.org/abs/2511.16778) | 2025 |  | [users-01/GCL-OT](https://github.com/users-01/GCL-OT) |
| [Generative-Contrastive Heterogeneous Graph Neural Network](https://arxiv.org/abs/2404.02810) | 2024 | [DOI](https://doi.org/10.1109/tbdata.2025.3570082) | [wangyu0627/GC-HGNN](https://github.com/wangyu0627/GC-HGNN) |
| [GEOM-GCN: Geometric Graph Convolutional Networks](https://arxiv.org/abs/2002.05287) | 2020 |  | [graphdml-uiuc-jlu/geom-gcn](https://github.com/graphdml-uiuc-jlu/geom-gcn) |
| [GLBench: A Comprehensive Benchmark for Graph with Large Language Models](https://arxiv.org/abs/2407.07457) | 2024 | [DOI](https://doi.org/10.52202/079017-1340) | [NineAbyss/GLBench](https://github.com/NineAbyss/GLBench) |
| [GMNN: Graph Markov Neural Networks](https://arxiv.org/abs/1905.06214) | 2019 |  | [DeepGraphLearning/GMNN](https://github.com/DeepGraphLearning/GMNN) |
| [GraFN: Semi-Supervised Node Classification on Graph with Few Labels via Non-Parametric Distribution Assignment](https://arxiv.org/abs/2204.01303) | 2022 | [DOI](https://doi.org/10.1145/3477495.3531838) | [Junseok0207/GraFN](https://github.com/Junseok0207/GraFN) |
| [Grain: Improving Data Efficiency of Graph Neural Networks via Diversified Influence Maximization](https://arxiv.org/abs/2108.00219) | 2021 |  | [zwt233/Grain](https://github.com/zwt233/Grain) |
| [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | 2018 |  | [PetarV-/GAT](https://github.com/PetarV-/GAT) |
| [Graph Contrastive Learning Meets Graph Meta Learning: A Unified Method for Few-shot Node Tasks](https://arxiv.org/abs/2309.10376) | 2023 | [DOI](https://doi.org/10.1145/3589334.3645367) | [Haoliu-cola/COLA](https://github.com/Haoliu-cola/COLA) |
| [Graph Contrastive Learning with Adaptive Augmentation](https://arxiv.org/abs/2010.14945) | 2021 | [DOI](https://doi.org/10.1145/3442381.3449802) | [CRIPAC-DIG/GCA](https://github.com/CRIPAC-DIG/GCA) |
| [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902) | 2020 |  | [Shen-Lab/GraphCL](https://github.com/Shen-Lab/GraphCL) |
| [Graph Few-shot Learning via Knowledge Transfer](https://arxiv.org/abs/1910.03053) | 2020 | [DOI](https://doi.org/10.1609/aaai.v34i04.6142) | [huaxiuyao/GFL](https://github.com/huaxiuyao/GFL) |
| [Graph Foundation Models: Concepts, Opportunities and Challenges](https://arxiv.org/abs/2310.11829) | 2023 | [DOI](https://doi.org/10.1109/tpami.2025.3548729) |  |
| [Graph Machine Learning in the Era of Large Language Models (LLMs)](https://arxiv.org/abs/2404.14928) | 2025 | [DOI](https://doi.org/10.1145/3732786) |  |
| [Graph Meta Learning via Local Subgraphs](https://arxiv.org/abs/2006.07889) | 2020 |  | [mims-harvard/G-Meta](https://github.com/mims-harvard/G-Meta) |
| [Graph Policy Network for Transferable Active Learning on Graphs](https://arxiv.org/abs/2006.13463) | 2020 |  | [ShengdingHu/GraphPolicyNetworkActiveLearning](https://github.com/ShengdingHu/GraphPolicyNetworkActiveLearning) |
| [Graph Prototypical Networks for Few-shot Learning on Attributed Networks](https://arxiv.org/abs/2006.12739) | 2020 | [DOI](https://doi.org/10.1145/3340531.3411922) | [kaize0409/GPN_Graph-Few-shot](https://github.com/kaize0409/GPN_Graph-Few-shot) |
| [Graph Representation Learning via Graphical Mutual Information Maximization](https://arxiv.org/abs/2002.01169) | 2020 | [DOI](https://doi.org/10.1145/3366423.3380112) | [zpeng27/GMI](https://github.com/zpeng27/GMI) |
| [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921) | 2025 | [DOI](https://doi.org/10.1145/3777378) |  |
| [Graph Self-Supervised Learning: A Survey](https://arxiv.org/abs/2103.00111) | 2022 | [DOI](https://doi.org/10.1109/tkde.2022.3172903) |  |
| [Graph Structure Learning for Robust Graph Neural Networks](https://arxiv.org/abs/2005.10203) | 2020 | [DOI](https://doi.org/10.1145/3394486.3403049) | [ChandlerBang/Pro-GNN](https://github.com/ChandlerBang/Pro-GNN) |
| [Graph World Model](https://arxiv.org/abs/2507.10539) | 2025 |  | [ulab-uiuc/GWM](https://github.com/ulab-uiuc/GWM) |
| [Graph World Models: Concepts, Taxonomy, and Future Directions](https://arxiv.org/abs/2604.27895) | 2026 |  |  |
| [GraphCLIP: Enhancing Transferability in Graph Foundation Models for Text-Attributed Graphs](https://arxiv.org/abs/2410.10329) | 2025 | [DOI](https://doi.org/10.1145/3696410.3714801) | [ZhuYun97/GraphCLIP](https://github.com/ZhuYun97/GraphCLIP) |
| [GraphEdit: Large Language Models for Graph Structure Learning](https://arxiv.org/abs/2402.15183) | 2024 |  | [HKUDS/GraphEdit](https://github.com/HKUDS/GraphEdit) |
| [GraphFM: A Comprehensive Benchmark for Graph Foundation Models](https://arxiv.org/abs/2406.08310) | 2024 |  | [NYUSHCS/GraphFM (stated in paper](https://github.com/NYUSHCS/GraphFM (stated in paper) |
| [GraphGPT: Graph Instruction Tuning for Large Language Models](https://arxiv.org/abs/2310.13023) | 2024 | [DOI](https://doi.org/10.1145/3626772.3657775) | [HKUDS/GraphGPT](https://github.com/HKUDS/GraphGPT) |
| [GraphLLM: Boosting Graph Reasoning Ability of Large Language Model](https://arxiv.org/abs/2310.05845) | 2023 | [DOI](https://doi.org/10.1109/tbdata.2025.3627488) | [mistyreed63849/Graph-LLM](https://github.com/mistyreed63849/Graph-LLM) |
| [GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner](https://arxiv.org/abs/2304.04779) | 2023 | [DOI](https://doi.org/10.1145/3543507.3583379) | [THUDM/GraphMAE2](https://github.com/THUDM/GraphMAE2) |
| [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://doi.org/10.1145/3534678.3539321) | 2022 |  | [THUDM/GraphMAE](https://github.com/THUDM/GraphMAE) |
| [GraphPrompt: Unifying Pre-Training and Downstream Tasks for Graph Neural Networks](https://arxiv.org/abs/2302.08043) | 2023 | [DOI](https://doi.org/10.1145/3543507.3583386) | [Starlien95/GraphPrompt](https://github.com/Starlien95/GraphPrompt) |
| [GRAPHTEXT: GRAPH REASONING IN TEXT SPACE](https://arxiv.org/abs/2310.01089) | 2023 |  | [AndyJZhao/GraphText](https://github.com/AndyJZhao/GraphText) |
| [Grenade: Graph-Centric Language Model for Self-Supervised Representation Learning on Text-Attributed Graphs](https://arxiv.org/abs/2310.15109) | 2023 | [DOI](https://doi.org/10.18653/v1/2023.findings-emnlp.181) | [bigheiniu/GRENADE](https://github.com/bigheiniu/GRENADE) |
| [Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning](https://arxiv.org/abs/2305.19523) | 2024 |  | [XiaoxinHe/TAPE](https://github.com/XiaoxinHe/TAPE) |
| [Hierarchical Graph Convolutional Networks for Semi-supervised Node Classification](https://arxiv.org/abs/1902.06667) | 2019 | [DOI](https://doi.org/10.24963/ijcai.2019/630) | [CRIPAC-DIG/H-GCN](https://github.com/CRIPAC-DIG/H-GCN) |
| [HopRank: Self-Supervised LLM Preference-Tuning on Graphs for Few-Shot Node Classification](https://arxiv.org/abs/2604.17271) | 2026 |  | [AlexandreWANG915/HopRank](https://github.com/AlexandreWANG915/HopRank) |
| [How Attentive Are Graph Attention Networks?](https://arxiv.org/abs/2105.14491) | 2022 |  | [tech-srl/how_attentive_are_gats](https://github.com/tech-srl/how_attentive_are_gats) |
| [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) | 2019 |  | [weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns) |
| [HyPAC: Cost-Efficient LLMs–Human Hybrid Annotation with PAC Error Guarantees](https://arxiv.org/abs/2602.02550) | 2026 |  | [https://anonymous.4open.science/r/HyPAC-B5AD](https://anonymous.4open.science/r/HyPAC-B5AD) |
| [IceBerg: Debiased Self-Training for Class-Imbalanced Node Classification](https://arxiv.org/abs/2502.06280) | 2025 | [DOI](https://doi.org/10.1145/3696410.3714963) | [ZhixunLEE/IceBerg](https://github.com/ZhixunLEE/IceBerg) |
| [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) | 2017 |  | [williamleif/GraphSAGE](https://github.com/williamleif/GraphSAGE) |
| [InfoGCL: Information-Aware Graph Contrastive Learning](https://arxiv.org/abs/2110.15438) | 2021 |  |  |
| [Information Gain Propagation: A New Way to Graph Active Learning with Soft Labels](https://arxiv.org/abs/2203.01093) | 2022 |  | [zwt233/IGP](https://github.com/zwt233/IGP) |
| [Informative Pseudo-Labeling for Graph Neural Networks with Few Labels](https://arxiv.org/abs/2201.07951) | 2022 | [DOI](https://doi.org/10.1007/s10618-022-00879-4) |  |
| [Interpretable Column Annotation with LLM-Symbolized Decision Process Materialization](https://arxiv.org/abs/2607.25228) | 2026 |  | [T-Lab/SymCA](https://github.com/T-Lab/SymCA) |
| [Interpretable-by-Design Text Understanding with Iteratively Generated Concept Bottleneck](https://arxiv.org/abs/2310.19660) | 2024 |  | [JMRLudan/TBM](https://github.com/JMRLudan/TBM) |
| [Item Tagging for Information Retrieval: A Tripartite Graph Neural Network based Approach](https://arxiv.org/abs/2008.11567) | 2020 | [DOI](https://doi.org/10.1145/3397271.3401438) | [kyriemao/TagGNN-SIGIR](https://github.com/kyriemao/TagGNN-SIGIR) |
| [Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings](https://arxiv.org/abs/2006.13009) | 2020 |  | [hugochan/IDGL](https://github.com/hugochan/IDGL) |
| [Joint learning of feature and topology for multi-view graph convolutional network](https://doi.org/10.1016/j.neunet.2023.09.006) | 2023 |  | [YuhongChen2320/JFGCN](https://github.com/YuhongChen2320/JFGCN) |
| [Label-Free Node Classification on Graphs with Large Language Models (LLMs)](https://arxiv.org/abs/2310.04668) | 2024 |  | [CurryTang/LLMGNN](https://github.com/CurryTang/LLMGNN) |
| [Large Knowledge Model: Perspectives and Challenges](https://arxiv.org/abs/2312.02706) | 2024 | [DOI](https://doi.org/10.3724/2096-7004.di.2024.0001) |  |
| [Large Language Models and Knowledge Graphs: Opportunities and Challenges](https://arxiv.org/abs/2308.06374) | 2023 | [DOI](https://doi.org/10.4230/tgdk.1.1.2) |  |
| [Large Language Models as Topological Structure Enhancers for Text-Attributed Graphs](https://arxiv.org/abs/2311.14324) | 2023 |  | [sunshy-1/LLM4GraphTopology](https://github.com/sunshy-1/LLM4GraphTopology) |
| [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) | 2024 |  |  |
| [Large Language Models on Graphs: A Comprehensive Survey](https://arxiv.org/abs/2312.02783) | 2024 | [DOI](https://doi.org/10.1109/tkde.2024.3469578) |  |
| [Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods](https://arxiv.org/abs/2110.14446) | 2021 |  | [CUAI/Non-Homophily-Large-Scale](https://github.com/CUAI/Non-Homophily-Large-Scale) |
| [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514) | 2022 |  | [nerdslab/bgrl](https://github.com/nerdslab/bgrl) |
| [Learnable Graph Convolutional Network and Feature Fusion for Multi-view Learning](https://arxiv.org/abs/2211.09155) | 2023 | [DOI](https://doi.org/10.1016/j.inffus.2023.02.013) |  |
| [Learning Ad Hoc Network Dynamics via Graph-Structured World Models](https://arxiv.org/abs/2604.14811) | 2026 |  | [cankaracelebi/WM-cluster (announced in PDF](https://github.com/cankaracelebi/WM-cluster (announced in PDF) |
| [Learning Discrete Structures for Graph Neural Networks](https://arxiv.org/abs/1903.11960) | 2019 |  | [lucfra/LDS-GNN](https://github.com/lucfra/LDS-GNN) |
| [Learning Knowledge Graph-based World Models of Textual Environments](https://arxiv.org/abs/2106.09608) | 2021 |  |  |
| [Learning on Large-Scale Text-Attributed Graphs via Variational Inference](https://arxiv.org/abs/2210.14709) | 2023 |  | [AndyJZhao/GLEM](https://github.com/AndyJZhao/GLEM) |
| [Learning Robust Representations via Multi-View Information Bottleneck](https://arxiv.org/abs/2002.07017) | 2020 |  | [mfederici/Multi-View-Information-Bottleneck](https://github.com/mfederici/Multi-View-Information-Bottleneck) |
| [Learning Strong Graph Neural Networks with Weak Information](https://arxiv.org/abs/2305.18457) | 2023 | [DOI](https://doi.org/10.1145/3580305.3599410) | [yixinliu233/D2PT](https://github.com/yixinliu233/D2PT) |
| [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | 2021 |  | [openai/CLIP](https://github.com/openai/CLIP) |
| [Let's Synthesize Step by Step: Iterative Dataset Synthesis with Large Language Models by Extrapolating Errors from Small Models](https://arxiv.org/abs/2310.13671) | 2023 |  | [RickySkywalker/Synthesis_Step-by-Step_Official](https://github.com/RickySkywalker/Synthesis_Step-by-Step_Official) |
| [Leveraging Large Language Models for Effective Label-free Node Classification in Text-Attributed Graphs](https://doi.org/10.1145/3726302.3730021) | 2025 |  | [HKBU-LAGAS/Locle](https://github.com/HKBU-LAGAS/Locle) |
| [Leveraging Large Language Models for Node Generation in Few-Shot Learning on Text-Attributed Graphs](https://arxiv.org/abs/2310.09872) | 2024 | [DOI](https://doi.org/10.1609/aaai.v39i12.33428) | [jianxiangyu/LLM4NG](https://github.com/jianxiangyu/LLM4NG) |
| [LLaGA: Large Language and Graph Assistant](https://arxiv.org/abs/2402.08170) | 2024 |  | [VITA-Group/LLaGA](https://github.com/VITA-Group/LLaGA) |
| [LOGIN: A Large Language Model Consulted Graph Neural Network Training Framework](https://doi.org/10.1145/3701551.3703488) | 2024 |  | [QiaoYRan/LOGIN](https://github.com/QiaoYRan/LOGIN) |
| [Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples](https://jmlr.org/papers/v7/belkin06a.html) | 2006 |  |  |
| [Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification](https://arxiv.org/abs/2009.03509) | 2021 | [DOI](https://doi.org/10.24963/ijcai.2021/214) | [PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp](https://github.com/PaddlePaddle/PGL/tree/main/ogb_examples/nodeproppred/unimp) |
| [Measuring Statistical Dependence with Hilbert-Schmidt Norms](https://doi.org/10.1007/11564089_7) | 2005 |  |  |
| [Meta-GNN: On Few-shot Node Classification in Graph Meta-learning](https://arxiv.org/abs/1905.09718) | 2019 |  | [AI-DL-Conference/Meta-GNN](https://github.com/AI-DL-Conference/Meta-GNN) |
| [Meta-GPS++: Enhancing Graph Meta-Learning with Contrastive Learning and Self-Training](https://arxiv.org/abs/2407.14732) | 2024 | [DOI](https://doi.org/10.1145/3679018) | [KEAML-JLU/Meta-GPS-Plus](https://github.com/KEAML-JLU/Meta-GPS-Plus) |
| [Mixed Graph Contrastive Network for Semi-Supervised Node Classification](https://arxiv.org/abs/2206.02796) | 2022 | [DOI](https://doi.org/10.1145/3641549) | [xihongyang1999/MGCN](https://github.com/xihongyang1999/MGCN) |
| [MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing](https://arxiv.org/abs/1905.00067) | 2019 |  | [samihaija/mixhop](https://github.com/samihaija/mixhop) |
| [Model Change Active Learning in Graph-Based Semi-supervised Learning](https://arxiv.org/abs/2110.07739) | 2024 |  | [millerk22/model-change-paper](https://github.com/millerk22/model-change-paper) |
| [Multi-graph Fusion Graph Convolutional Networks with pseudo-label supervision](https://doi.org/10.1016/j.neunet.2022.11.027) | 2023 |  |  |
| [Multi-Scale Spectral Selection and Entropy-Guided Uncertainty Fusion for Multimodal Rumor Detection](https://doi.org/10.18653/v1/2026.findings-acl.55) | 2026 |  |  |
| [Neighborhood Homophily-based Graph Convolutional Network](https://arxiv.org/abs/2301.09851) | 2023 | [DOI](https://doi.org/10.1145/3583780.3615195) | [rockcor/NHGCN](https://github.com/rockcor/NHGCN) |
| [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | 2016 | [DOI](https://doi.org/10.18653/v1/p16-1162) | [rsennrich/subword-nmt](https://github.com/rsennrich/subword-nmt) |
| [Neural networks and physical systems with emergent collective computational abilities](https://doi.org/10.1073/pnas.79.8.2554) | 1982 |  |  |
| [Next Generation Active Learning: Mixture of LLMs in the Loop](https://arxiv.org/abs/2601.15773) | 2026 |  | [qijindou/MoLLIA](https://github.com/qijindou/MoLLIA) |
| [Node Feature Extraction by Self-Supervised Multi-Scale Neighborhood Prediction](https://arxiv.org/abs/2111.00064) | 2022 |  | [amzn/pecos/tree/mainline/examples/giant-xrt](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt) |
| [Node Similarity Preserving Graph Convolutional Networks](https://arxiv.org/abs/2011.09643) | 2021 | [DOI](https://doi.org/10.1145/3437963.3441735) | [ChandlerBang/SimP-GCN](https://github.com/ChandlerBang/SimP-GCN) |
| [Normalize Then Propagate: Efficient Homophilous Regularization for Few-shot Semi-Supervised Node Classification](https://arxiv.org/abs/2501.08581) | 2025 |  | [Pallaksch/NormProp](https://github.com/Pallaksch/NormProp) |
| [Open Graph Benchmark: Datasets for Machine Learning on Graphs](https://arxiv.org/abs/2005.00687) | 2020 |  | [snap-stanford/ogb](https://github.com/snap-stanford/ogb) |
| [Open-World Graph Active Learning for Node Classification](https://doi.org/10.1145/3607144) | 2024 |  |  |
| [Optimistic World Models: Efficient Exploration in Model-Based Deep Reinforcement Learning](https://arxiv.org/abs/2602.10044) | 2026 |  |  |
| [PATTON: Language Model Pretraining on Text-Rich Networks](https://arxiv.org/abs/2305.12268) | 2023 | [DOI](https://doi.org/10.18653/v1/2023.acl-long.387) | [PeterGriffinJin/Patton](https://github.com/PeterGriffinJin/Patton) |
| [Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/abs/1811.05868) | 2019 |  | [shchur/gnn-benchmark](https://github.com/shchur/gnn-benchmark) |
| [ProG: A Graph Prompt Learning Benchmark](https://arxiv.org/abs/2406.05346) | 2024 |  | [sheldonresearch/ProG](https://github.com/sheldonresearch/ProG) |
| [PROGEN: Progressive Zero-shot Dataset Generation via In-context Feedback](https://arxiv.org/abs/2210.12329) | 2022 | [DOI](https://doi.org/10.18653/v1/2022.findings-emnlp.269) | [HKUNLP/ProGen](https://github.com/HKUNLP/ProGen) |
| [Query-driven Active Surveying for Collective Classification](http://people.cs.vt.edu/~bhuang/papers/namata-mlg12.pdf) | 2012 |  |  |
| [Realistic Evaluation of Deep Semi-Supervised Learning Algorithms](https://arxiv.org/abs/1804.09170) | 2019 |  | [brain-research/realistic-ssl-evaluation](https://github.com/brain-research/realistic-ssl-evaluation) |
| [Relation-aware Heterogeneous Graph for User Profiling](https://arxiv.org/abs/2110.07181) | 2021 | [DOI](https://doi.org/10.1145/3459637.3482170) | [CRIPAC-DIG/RHGN](https://github.com/CRIPAC-DIG/RHGN) |
| [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) | 2019 |  |  |
| [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) | 2023 |  |  |
| [Revisiting Heterophily For Graph Neural Networks](https://arxiv.org/abs/2210.07606) | 2022 |  | [SitaoLuan/ACM-GNN](https://github.com/SitaoLuan/ACM-GNN) |
| [Revisiting K-mer Profile for Effective and Scalable Genome Representation Learning](https://arxiv.org/abs/2411.02125) | 2024 | [DOI](https://doi.org/10.52202/079017-3778) | [abdcelikkanat/revisitingkmers](https://github.com/abdcelikkanat/revisitingkmers) |
| [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861) | 2016 |  | [kimiyoung/planetoid](https://github.com/kimiyoung/planetoid) |
| [RIM: Reliable Influence-based Active Learning on Graphs](https://arxiv.org/abs/2110.14854) | 2021 |  | [zwt233/RIM](https://github.com/zwt233/RIM) |
| [Self-Supervised Learning from a Multi-View Perspective](https://arxiv.org/abs/2006.05576) | 2021 |  | [yaohungt/Self_Supervised_Learning_Multiview](https://github.com/yaohungt/Self_Supervised_Learning_Multiview) |
| [Semantic Graph Neural Network with Multi-measure Learning for Semi-supervised Classification](https://arxiv.org/abs/2212.01749) | 2025 |  | [landrarwolf/ML-SGNN](https://github.com/landrarwolf/ML-SGNN) |
| [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | 2017 |  | [tkipf/gcn](https://github.com/tkipf/gcn) |
| [Semi-supervised Instruction Tuning for Large Language Models on Text-Attributed Graphs](https://arxiv.org/abs/2601.12807) | 2026 |  |  |
| [Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions](https://scholar.google.com/scholar?q=Semi-Supervised%20Learning%20Using%20Gaussian%20Fields%20and%20Harmonic%20Functions) | 2003 |  |  |
| [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) | 2019 | [DOI](https://doi.org/10.18653/v1/d19-1410) | [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) |
| [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | 2014 |  |  |
| [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414) | 2019 |  | [google-research/google-research/tree/master/representation_similarity](https://github.com/google-research/google-research/tree/master/representation_similarity) |
| [Simple Unsupervised Graph Representation Learning](https://doi.org/10.1609/aaai.v36i7.20748) | 2022 |  | [YujieMo/SUGRL](https://github.com/YujieMo/SUGRL) |
| [SimTeG: A Frustratingly Simple Approach Improves Textual Graph Learning](https://arxiv.org/abs/2308.02565) | 2023 |  | [vermouthdky/SimTeG](https://github.com/vermouthdky/SimTeG) |
| [SLAPS: Self-Supervision Improves Structure Learning for Graph Neural Networks](https://arxiv.org/abs/2102.05034) | 2021 |  | [BorealisAI/SLAPS-GNN](https://github.com/BorealisAI/SLAPS-GNN) |
| [Spatio-Temporal Data Mining for Climate Data: Advances, Challenges, and Opportunities](https://doi.org/10.1007/978-3-642-40837-3_3) | 2013 |  |  |
| [Statistical Modeling: The Two Cultures](https://doi.org/10.1214/ss/1009213726) | 2001 |  |  |
| [Supervised Feature Selection via Dependence Estimation](https://doi.org/10.1145/1273496.1273600) | 2007 |  | [http://elefant.developer.nicta.com.au (Elefant package, footnote 6](http://elefant.developer.nicta.com.au (Elefant package, footnote 6) |
| [Supervised Graph Contrastive Learning for Few-shot Node Classification](https://arxiv.org/abs/2203.15936) | 2022 | [DOI](https://doi.org/10.1007/978-3-031-26390-3_24) |  |
| [SYNAPSE-G: Bridging Large Language Models and Graph Learning for Rare Event Classification](https://arxiv.org/abs/2508.09544) | 2025 |  |  |
| [Task Contamination: Language Models May Not Be Few-Shot Anymore](https://arxiv.org/abs/2312.16337) | 2023 | [DOI](https://doi.org/10.1609/aaai.v38i16.29808) |  |
| [Task-Adaptive Few-shot Node Classification](https://arxiv.org/abs/2206.11972) | 2022 | [DOI](https://doi.org/10.1145/3534678.3539265) | [SongW-SW/TENT](https://github.com/SongW-SW/TENT) |
| [Task-Equivariant Graph Few-shot Learning](https://arxiv.org/abs/2305.18758) | 2023 | [DOI](https://doi.org/10.1145/3580305.3599515) | [sung-won-kim/TEG](https://github.com/sung-won-kim/TEG) |
| [TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://doi.org/10.14778/3665844.3665863) | 2024 |  | [decisionintelligence/TFB](https://github.com/decisionintelligence/TFB) |
| [The good, the bad, and the ugly in chemical and biological data for machine learning](https://doi.org/10.1016/j.ddtec.2020.07.001) | 2020 |  |  |
| [Towards Unsupervised Deep Graph Structure Learning](https://arxiv.org/abs/2201.06367) | 2022 | [DOI](https://doi.org/10.1145/3485447.3512186) | [GRAND-Lab/SUBLIME](https://github.com/GRAND-Lab/SUBLIME) |
| [Transductive Linear Probing: A Novel Framework for Few-Shot Node Classification](https://arxiv.org/abs/2212.05606) | 2022 |  | [Zhen-Tan-dmml/TLP-FSNC](https://github.com/Zhen-Tan-dmml/TLP-FSNC) |
| [Uncertainty aware training to improve deep learning model calibration for classification of cardiac MR images](https://arxiv.org/abs/2308.15141) | 2023 | [DOI](https://doi.org/10.1016/j.media.2023.102861) | [tareend/Uncertainty-Aware-Training](https://github.com/tareend/Uncertainty-Aware-Training) |
| [Uncertainty for Active Learning on Graphs](https://arxiv.org/abs/2405.01462) | 2024 |  | [dfuchsgruber/uq-for-al-on-graphs](https://github.com/dfuchsgruber/uq-for-al-on-graphs) |
| [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) | 2020 |  | [SsnL/align_uniform](https://github.com/SsnL/align_uniform) |
| [Understanding Rollout Error in Graph World Models](https://arxiv.org/abs/2606.27780) | 2026 |  | [Hik289/graph_world_model_accumulative_error](https://github.com/Hik289/graph_world_model_accumulative_error) |
| [Unifying Generation and Prediction on Graphs with Latent Graph Diffusion](https://arxiv.org/abs/2402.02518) | 2024 | [DOI](https://doi.org/10.52202/079017-1979) | [zhouc20/LatentGraphDiffusion](https://github.com/zhouc20/LatentGraphDiffusion) |
| [Unifying Large Language Models and Knowledge Graphs: A Roadmap](https://arxiv.org/abs/2306.08302) | 2023 | [DOI](https://doi.org/10.1109/tkde.2024.3352100) |  |
| [UPL: Uncertainty-aware Pseudo-labeling for Imbalance Transductive Node Classification](https://arxiv.org/abs/2502.00716) | 2025 |  |  |
| [Value Memory Graph: A Graph-Structured World Model for Offline Reinforcement Learning](https://arxiv.org/abs/2206.04384) | 2023 |  | [TsuTikgiau/ValueMemoryGraph](https://github.com/TsuTikgiau/ValueMemoryGraph) |
| [Verbalized Graph Representation Learning: A Fully Interpretable Graph Model Based on Large Language Models Throughout the Entire Process](https://arxiv.org/abs/2410.01457) | 2024 |  | [https://anonymous.4open.science/r/VGRL-7E1E](https://anonymous.4open.science/r/VGRL-7E1E) |
| [Verbalized Machine Learning: Revisiting Machine Learning with Language Models](https://arxiv.org/abs/2406.04344) | 2024 |  | [timxzz/VML_Examples](https://github.com/timxzz/VML_Examples) |
| [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906) | 2022 |  | [facebookresearch/vicreg](https://github.com/facebookresearch/vicreg) |
| [Visualizing Data using t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html) | 2008 |  | [https://lvdmaaten.github.io/tsne](https://lvdmaaten.github.io/tsne/) |
| [What Makes for Good Views for Contrastive Learning?](https://arxiv.org/abs/2005.10243) |  |  |  |
| [When Contrastive Learning Meets Active Learning: A Novel Graph Active Learning Paradigm with Self-Supervision](https://arxiv.org/abs/2010.16091) | 2020 |  |  |
| [When Do LLMs Help With Node Classification? A Comprehensive Analysis](https://arxiv.org/abs/2502.00829) | 2025 |  | [WxxShirley/LLMNodeBed](https://github.com/WxxShirley/LLMNodeBed) |
| [When Structure Doesn't Help: LLMs Do Not Read Text-Attributed Graphs as Effectively as We Expected](https://arxiv.org/abs/2511.16767) | 2025 |  | [hxu105/llm-graph](https://github.com/hxu105/llm-graph) |
| [Where LLM Annotators Fail: Label-Free Learning on Graphs with LLMs](https://arxiv.org/abs/2605.27913) | 2026 |  | [thapaliya19/CANE](https://github.com/thapaliya19/CANE) |
| [Where to Mask: Structure-Guided Masking for Graph Masked Autoencoders](https://arxiv.org/abs/2404.15806) | 2024 | [DOI](https://doi.org/10.24963/ijcai.2024/241) | [LiuChuang0059/StructMAE](https://github.com/LiuChuang0059/StructMAE) |
| [Who Should Teach? Confidence-Aware Dual-Teacher Learning for Few-Shot Node Classification on Text-Attributed Graphs](https://arxiv.org/abs/2608.22127) | 2026 |  | [https://buly.kr/GP4yN34 (URL shortener](https://buly.kr/GP4yN34 (URL shortener) |
| [World Model for Robot Learning: A Comprehensive Survey](https://arxiv.org/abs/2605.00080) | 2026 |  |  |
| [ZeroG: Investigating Cross-dataset Zero-shot Transferability in Graphs](https://arxiv.org/abs/2402.11235) | 2024 | [DOI](https://doi.org/10.1145/3637528.3671982) | [NineAbyss/ZeroG](https://github.com/NineAbyss/ZeroG) |

## Adding a paper

Add a row to Part 2 with the arXiv or DOI link. When the paper enters a discussion or an experiment, add it to Part 1
under the link of the argument it bears on, with one line on why it matters. Keep PDFs in the shared reference library, not in git.
