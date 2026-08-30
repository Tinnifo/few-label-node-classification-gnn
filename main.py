"""Hydra experiment interface: conf/ is the protocol, one arm = one `semantic=` option.

Usage:
  python main.py                                             # A0 on cora
  python main.py dataset=pubmed semantic=sbert               # one arm
  python main.py -m dataset=cora,pubmed,citeseer_tag semantic=none,sbert,e5,gpt3l
  python main.py -m dataset=cora,pubmed,citeseer_tag semantic=tape,tape_leak
  python main.py training.patience=300 'seeds=[0,1,2]'       # ad-hoc overrides

The argparse CLI in src/cg3_semantic.py still works for one-off debugging, but
its defaults are DEBUG defaults (early stopping off, patience 50). conf/ is the
experiment protocol (early stopping on, patience 200, identical on every arm)
— run experiments through this entrypoint only.
"""

from __future__ import annotations

import argparse

import hydra
from omegaconf import DictConfig

from src.cg3_semantic import main as run_arm


def to_namespace(cfg: DictConfig) -> argparse.Namespace:
    """Flatten the composed config into the namespace src/cg3_semantic.py expects."""
    sem, s, tr, out = cfg.semantic, cfg.structure, cfg.training, cfg.output
    needs_texts = bool(sem.enabled) and not sem.embeddings
    return argparse.Namespace(
        # data
        dataset=cfg.dataset.name,
        data_root=cfg.dataset.data_root,
        texts=cfg.dataset.texts if needs_texts else None,
        # label budget
        label_strategy=cfg.label_strategy,
        budget=cfg.budget,
        seeds=",".join(str(x) for x in cfg.seeds),
        # structure (CG3)
        local_model=s.local_model, global_model=s.global_model,
        hidden_local=s.hidden_local, hidden_global=s.hidden_global,
        coarsen_level=s.coarsen_level, max_node_wgt=s.max_node_wgt,
        channel_num=s.channel_num, node_wgt_embed_dim=s.node_wgt_embed_dim,
        temperature=s.temperature, hp1=s.hp1,
        # semantic view
        semantic=bool(sem.enabled),
        semantic_dim=sem.dim, semantic_hidden=sem.hidden,
        descriptor=sem.descriptor, class_names=sem.class_names,
        keep_label_leak=bool(sem.keep_label_leak),
        descriptor_model=sem.descriptor_model, descriptor_max_tokens=sem.descriptor_max_tokens,
        encoder_model=sem.encoder, semantic_embeddings=sem.embeddings,
        fusion=cfg.fusion,
        hsic_threshold=cfg.hsic.threshold, hsic_sigma=cfg.hsic.sigma,
        hsic_weight=cfg.hsic.weight, hsic_max_samples=cfg.hsic.max_samples,
        # training
        epochs=tr.epochs, lr=tr.lr, weight_decay=tr.weight_decay, dropout=tr.dropout,
        early_stopping=bool(tr.early_stopping), patience=tr.patience, device=tr.device,
        # output
        output=out.dir, epoch_log_every=out.epoch_log_every,
        no_mlflow=not out.mlflow, mlflow_uri=out.mlflow_uri,
        mlflow_experiment=out.mlflow_experiment,
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def app(cfg: DictConfig) -> float:
    # returns mean test accuracy — the objective a Hydra sweeper optimizes
    return run_arm(to_namespace(cfg))


if __name__ == "__main__":
    app()
