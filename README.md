# tiny-jev

**English** | [中文](README.zh-CN.md)

A tiny, from-scratch reimplementation of the ideas behind **Jev**-style decision models, small enough to train and run on a laptop. It is built for teaching: every claim is backed by a runnable experiment.

The core idea: instead of having an LLM **generate** an action token by token, put each candidate action **into the input** and have the model output **a probability for every candidate at once**:

```
f(observation, goal, action_i)  →  vector z_i  →  logit  →  softmax  →  p_i
```

![One complete episode, step by step](out/scenario.gif)

> This is an independent educational project. Jev's internals are not public. **NanoJev** and **Laya** are community open-source implementations of these ideas; the two architectures here follow their structure, but neither they nor this repo should be taken as Jev's actual implementation.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python experiments.py      # 6 experiments → out/*.png  (uses the small checkpoints in the repo)
.venv/bin/python scenario.py         # one episode, input → reasoning → output → out/scenario.gif
```

The pretrained-encoder models (Qwen3-0.6B / ModernBERT-base) produce fine-tuned weights of 270 MB / 600 MB, too large for GitHub. Recreate everything (about 40 minutes on an Apple Silicon GPU; downloads ~2 GB of base weights from Hugging Face):

```bash
./run_all.sh
```

## The environment: TinyFS

A mini "file-system agent" task, following the open file / list files example:

- **State**: which files exist, which ones the agent already knows about, which one is open. Some files are **locked**: opening them fails until they are unlocked. The agent **cannot see** whether a file is locked.
- **Goal**: `edit` / `read` / `delete` a file.
- **Candidates**: mapped from the task and the current state (`list_files`, `open_file X`, `unlock_file X`, `edit_file X`, …), 2 to 22 per state.
- **Target probabilities p\***: a rule-based expert's distribution over the candidates, e.g. when the target file is not known yet: `list_files` 0.7, `search` 0.3.

## Models

| Model | What it is | Params |
|---|---|---|
| `gpt` | Baseline: generates the action text token by token | 0.7 M |
| `nanojev` | Encodes each `(obs, goal, action_i)` separately → vector z_i → attention across candidates → logit | 0.7 M |
| `laya` | Encodes `[CLS] goal [SEP] obs [MASK] a1 [MASK] a2 …` in **one** bidirectional pass, reads the vector at each `[MASK]` | 0.7 M |
| `nanojev_qwen` | NanoJev structure on top of **Qwen3-0.6B** (last 4 layers + head fine-tuned) | 600 M (67 M trained) |
| `laya_bert` | Laya structure on top of **ModernBERT-base** (fully fine-tuned) | 149 M |

All Jev-style models share the same output interface: one logit per candidate, softmax over candidates.

## Results

All numbers below come from actually training and running the models in this repo.

| # | Experiment | Result |
|---|---|---|
| 1 | **Speed**: generate vs. score | GPT latency grows linearly with action length (32 tokens: 56 ms). Jev scores all candidates in one forward pass. Real tasks: **10.3 ms vs 2.2 ms** |
| 2 | **Exploration**: locked file | Same model. Always picking the top action: **0%** success (repeats `open_file` forever). Sampling from the probabilities: **68%** (eventually tries `unlock_file`) |
| 3 | **Inside the model** | candidates → vectors → attention → logits → probabilities **0.80 / 0.15 / 0.05**, matching the expert's targets |
| 4 | **Cross-entropy vs Brier** | Both learn the target probabilities (calibration plots). CE converged faster here (test KL 0.017 vs 0.159); Brier's gradient vanishes when a prediction is badly wrong |
| 5 | **NanoJev vs Laya cost** | With 22 candidates, NanoJev processes **826 tokens** per decision (obs + goal repeated for every candidate), Laya **175** |
| 6 | **Why reuse a pretrained model** | Goals with unseen synonyms (modify / view / remove): from-scratch models 55–75%, **Qwen3 92%**. ModernBERT did **not** help here (71%). The GPT baseline invented **non-existent actions 63%** of the time on unseen file names; Jev models can only choose valid candidates |
| 7 | **One full episode** | `list_files` → `open_file` (fails, locked) → `unlock_file` (sampled at p = 0.17) → `open_file` → `edit_file` ✓ |

![Speed](out/1_speed.png)
![Exploration](out/2_explore.png)
![Inside the model](out/3_inside.png)
![Cross-entropy vs Brier](out/4_loss.png)
![NanoJev vs Laya](out/5_cost.png)
![Pretrained vs from scratch](out/6_generalize.png)

## Caveats

- Cross-entropy and Brier can both learn target probabilities; calibration does not inherently require switching to RL. Brier is used here as a classic probability score, not as Jev's published recipe.
- Candidate **vectors** z, scalar **logits**, and **probabilities** after softmax are different things; the experiments plot each separately.
- In experiment 7, every action is sampled by the model; the shown episode was selected from several runs because it contains a failure, an unlock and a success.
- Scale is tiny and the environment is synthetic. Results show the mechanisms, not production-level numbers.

## Files

| File | Contents |
|---|---|
| `env.py` | TinyFS: states, candidate mapping, expert targets p\*, dynamics |
| `data.py` | Word-level vocab, batching for NanoJev / Laya / GPT |
| `models.py` | The five models, cross-entropy and Brier losses |
| `train.py` | Training + evaluation on 3 test sets (same distribution, unseen synonyms, unseen file names) |
| `experiments.py` | Experiments 1–6 |
| `scenario.py` | Experiment 7: one episode, step by step |

Code comments are in Chinese.
