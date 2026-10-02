"""
把环境状态变成模型输入。

从零训练的模型用一个很小的词级分词器（词表只来自训练数据，没见过的词变成 [UNK]）；
预训练模型用它们自己的分词器（在 models.py 里）。
"""

import re

import torch

from env import State, candidates, expert

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
PAD, UNK, CLS, SEP, MASK, BOS, EOS = range(len(SPECIALS))


def words(text: str) -> list:
    """report.md → report . md ；open_file → open _ file"""
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower())


class Vocab:
    def __init__(self, states: list):
        seen = {"goal": len(SPECIALS)}            # context_ids 里固定加的前缀词
        for s in states:
            for t in [s.goal_text, s.obs_text] + candidates(s):
                for w in words(t):
                    seen.setdefault(w, len(SPECIALS) + len(seen))
        self.itos = SPECIALS + list(seen)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text: str) -> list:
        return [self.stoi.get(w, UNK) for w in words(text)]

    def __len__(self):
        return len(self.itos)


def targets(states: list, device) -> tuple:
    """目标概率 p*，以及哪些候选位置有效（不同状态的候选数量不同，要补齐）。"""
    K = max(len(candidates(s)) for s in states)
    p = torch.zeros(len(states), K)
    valid = torch.zeros(len(states), K, dtype=torch.bool)
    for b, s in enumerate(states):
        for k, prob in enumerate(expert(s).values()):
            p[b, k] = prob
            valid[b, k] = True
    return p.to(device), valid.to(device)


def context_ids(vocab: Vocab, s: State) -> list:
    return [CLS] + vocab.encode("goal " + s.goal_text) + [SEP] + vocab.encode(s.obs_text)


def pad(seqs: list, device) -> torch.Tensor:
    L = max(len(x) for x in seqs)
    return torch.tensor([x + [PAD] * (L - len(x)) for x in seqs], device=device)


def batch_nanojev(vocab: Vocab, states: list, device) -> dict:
    """NanoJev：每个候选一条序列 [CLS] goal [SEP] obs [SEP] action —— goal 和 obs 会被重复编码。"""
    seqs, owner = [], []
    for b, s in enumerate(states):
        ctx = context_ids(vocab, s)
        for a in candidates(s):
            seqs.append(ctx + [SEP] + vocab.encode(a))
            owner.append(b)
    return {"ids": pad(seqs, device), "owner": torch.tensor(owner, device=device)}


def batch_laya(vocab: Vocab, states: list, device) -> dict:
    """Laya：一个状态一条序列 [CLS] goal [SEP] obs [MASK] a1 [MASK] a2 …，记下每个 [MASK] 的位置。"""
    seqs, pos = [], []
    for s in states:
        ids = context_ids(vocab, s)
        p = []
        for a in candidates(s):
            p.append(len(ids))
            ids = ids + [MASK] + vocab.encode(a)
        seqs.append(ids)
        pos.append(p)
    K = max(len(p) for p in pos)
    return {"ids": pad(seqs, device), "mask_pos": torch.tensor([p + [0] * (K - len(p)) for p in pos], device=device)}


def batch_gpt(vocab: Vocab, states: list, actions: list, device) -> dict:
    """GPT 基线：[BOS] goal [SEP] obs [SEP] action [EOS]，只在 action 部分算 loss。"""
    seqs, starts = [], []
    for s, a in zip(states, actions):
        prefix = [BOS] + context_ids(vocab, s)[1:] + [SEP]
        seqs.append(prefix + vocab.encode(a) + [EOS])
        starts.append(len(prefix))
    ids = pad(seqs, device)
    loss_mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, (st, sq) in enumerate(zip(starts, seqs)):
        loss_mask[i, st:len(sq)] = True       # 预测 action 和 [EOS] 的位置
    return {"ids": ids, "loss_mask": loss_mask}
