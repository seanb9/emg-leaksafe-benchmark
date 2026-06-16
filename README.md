# Leak-Safe Within-User Benchmark for Compact Surface-EMG Grasp Decoding

Reproduction code for the paper *"A Leak-Safe Within-User Benchmark for Compact
Surface-EMG Grasp Decoding, with a Causal Sequence-Reasoning Decoder"*
(Barrett & Hartley, ReAble Labs).

This repository contains the **evaluation pipeline and models** needed to reproduce
the results in the paper from the public NinaPro DB2 dataset. It is a research
reproducibility artifact, not a product. See `NOTICE` for scope.

## What's here

- `emgv2/` — the pipeline: leak-safe windowing and segment splits (`data/`), the
  compact decoder and baselines (`models/`), the causal sequence decoder and metrics
  (`eval/`), and training (`train/`).
- `scripts/` — one entry point per experiment.
- `configs/` — exact run configurations for the reported experiments.

## What's **not** here

Trained model weights, cached data, the manuscript, and any ReAble Labs hardware,
firmware, or clinical product code. All results are reproduced from public data by
re-running the pipeline (see `NOTICE`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.12; PyTorch 2.12, NumPy 2.4, SciPy 1.17, scikit-learn 1.9.
Runs on CPU, CUDA, or Apple MPS.

## Data

Download **NinaPro DB2** (40 subjects, free, registration required) from
<http://ninapro.hevs.ch> and place the per-subject `.mat` files where the config's
`data_root` points. Then build the windowed cache:

```bash
python scripts/build_cache.py configs/db2_reable_hand.yaml      # 7-class within-user
python scripts/build_cache.py configs/db2_reable_10class.yaml   # 10-class LOSO
```

## Reproducing the paper

```bash
# Within-user (Table I, Table III, Figs 1–5): per-window / per-execution / false-act,
# classical TD+LDA baseline, matched-vote and ablation comparisons with significance.
python scripts/run_within_user.py configs/db2_reable_hand.yaml

# Ablations: no self-supervised pretraining; scaled-likelihood emission.
python scripts/run_within_user.py configs/db2_reable_hand_nossl.yaml --reuse-base
python scripts/run_within_user.py configs/db2_reable_hand_priorcorr.yaml --reuse-base

# Cross-subject 40-fold LOSO (Table II). Parallelise across workers with --test-subjects.
python scripts/run_loso_full.py configs/db2_reable_10class.yaml --size S \
    --supcon-epochs 20 --ft-epochs 60 --test-subjects 1-10

# Figures, and the synthetic positive-control.
python scripts/make_paper_figures.py
python scripts/run_synthetic.py
```

All runs use a fixed seed (1337). Per-fold LOSO results are checkpointed and the run
is resumable.

## Citation

```bibtex
@article{barrett2026leaksafe,
  title  = {A Leak-Safe Within-User Benchmark for Compact Surface-EMG Grasp
            Decoding, with a Causal Sequence-Reasoning Decoder},
  author = {Barrett, Sean and Hartley, William},
  year   = {2026},
  note   = {ReAble Labs}
}
```

## License

MIT — see `LICENSE`. The dataset (NinaPro DB2) is governed by its own terms.
