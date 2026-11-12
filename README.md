# RL for Compiler Pass Ordering (LLVM Code-Size Reduction)

Proximal Policy Optimization (PPO) agent for selecting per-function LLVM optimization pass sequences to reduce IR instruction count relative to `-Oz`.

## Setup & Evaluation

- **Corpus**: 4,000 C functions (3,200 train / 800 test) represented via 56-dim Autophase features.
- **Action Space**: 16 scalar and loop optimization passes (`mem2reg`, `instcombine`, `simplifycfg`, `gvn`, `sccp`, `adce`, etc.), episode length = 12 steps.
- **Baselines**: `-Oz` default pipeline (1 compilation/func), Genetic Algorithm (500 compilations/func), and Random Search (500 compilations/func).
- **Compiler Drift**: Evaluates zero-shot transfer and fine-tuning from LLVM 15 to LLVM 17.

## Results Summary

| Method | LLVM Version | Budget (Comps/Func) | Geomean Size Ratio vs `-Oz` |
|---|---|---|---|
| `-Oz` (Reference) | LLVM 15 | 1 | 1.000 |
| PPO Agent | LLVM 15 | 1 | 0.992 |
| Genetic Algorithm | LLVM 15 | 500 | 0.998 |
| Random Search | LLVM 15 | 500 | 1.001 |
| PPO (Zero-Shot from LLVM 15) | LLVM 17 | 1 | 0.998 |
| PPO (Fine-Tuned on LLVM 17) | LLVM 17 | 1 | 0.993 |
| Genetic Algorithm Re-run | LLVM 17 | 500 | 0.997 |

## Running

```bash
pip install -r requirements.txt
python train_ppo.py
python baselines.py
python evaluate_and_analyze.py
```
