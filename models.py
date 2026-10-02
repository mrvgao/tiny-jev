"""
五个模型，对应视频里的三条路线：

    GPTPolicy      逐 token 生成动作文本（视频开头回顾的 GPT 路线），用来做速度和行为对比
    NanoJev        f(obs, goal, action_i) → 向量 z_i → 候选之间 attention → logit → softmax
    Laya           [CLS] obs goal [MASK] a1 [MASK] a2 … 一次双向编码，取每个 [MASK] 的向量 → logit → softmax
    NanoJevQwen    NanoJev 的结构，编码器换成预训练的 Qwen3-0.6B
    LayaBert       Laya 的结构，编码器换成预训练的 ModernBERT-base

Jev 类模型的输出接口都一样：每个候选一个 logit（实数），softmax 后是候选上的概率分布。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD


# ─────────────────────────────── 从零训练的小 Transformer ───────────────────────────────


class Encoder(nn.Module):
    def __init__(self, vocab_size, dim=128, depth=3, heads=4, max_len=512, causal=False):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.causal = causal

    def forward(self, ids):
        L = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=ids.device) if self.causal else None
        x = self.layers(x, mask=mask, src_key_padding_mask=ids == PAD, is_causal=self.causal)
        return self.norm(x)


class GPTPolicy(nn.Module):
    """GPT 路线：给定 [BOS] goal obs [SEP]，一个 token 一个 token 地写出动作。"""

    def __init__(self, vocab_size, dim=128, depth=3, heads=4):
        super().__init__()
        self.enc = Encoder(vocab_size, dim, depth, heads, causal=True)
        self.lm = nn.Linear(dim, vocab_size)

    def forward(self, ids):
        return self.lm(self.enc(ids))


class SetHead(nn.Module):
    """候选向量 z_i 之间做一次 attention（融合候选之间的关系），再线性变换成一个 logit。"""

    def __init__(self, dim, heads=4, set_attention=True):
        super().__init__()
        self.set_attention = set_attention
        if set_attention:
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, 1)  # 视频里的 Wx + b

    def forward(self, z, valid, return_attn=False):
        attn = None
        if self.set_attention:
            h, attn = self.attn(z, z, z, key_padding_mask=~valid, need_weights=True)
            z = self.norm(z + h)
        logits = self.out(z).squeeze(-1).masked_fill(~valid, float("-inf"))
        return (logits, attn) if return_attn else logits


def group(vectors, owner, B):
    """把 (所有候选, dim) 按所属状态分组成 (B, K, dim)，返回 valid 掩码。"""
    counts = torch.bincount(owner, minlength=B)
    K = int(counts.max())
    out = vectors.new_zeros(B, K, vectors.shape[-1])
    valid = torch.zeros(B, K, dtype=torch.bool, device=vectors.device)
    idx = torch.cat([torch.arange(int(c), device=vectors.device) for c in counts])
    out[owner, idx] = vectors
    valid[owner, idx] = True
    return out, valid


class NanoJev(nn.Module):
    def __init__(self, vocab_size, dim=128, depth=3, heads=4):
        super().__init__()
        self.enc = Encoder(vocab_size, dim, depth, heads)
        self.head = SetHead(dim, heads)

    def forward(self, batch, B, return_parts=False):
        z = self.enc(batch["ids"])[:, 0]                 # 每个候选序列的 [CLS] 向量 = z_i
        z, valid = group(z, batch["owner"], B)
        logits, attn = self.head(z, valid, return_attn=True)
        return (logits, {"z": z, "attn": attn, "valid": valid}) if return_parts else logits


class Laya(nn.Module):
    def __init__(self, vocab_size, dim=128, depth=3, heads=4):
        super().__init__()
        self.enc = Encoder(vocab_size, dim, depth, heads)
        self.head = SetHead(dim, heads, set_attention=False)  # 编码时候选之间已经交换过信息，不再需要集合 attention

    def forward(self, batch, B, return_parts=False):
        h = self.enc(batch["ids"])
        pos = batch["mask_pos"]
        z = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))  # 取出每个 [MASK] 位置的向量
        valid = pos > 0                                   # 补齐的位置记成 0（[CLS] 的位置，永远不是 [MASK]）
        logits = self.head(z, valid)
        return (logits, {"z": z, "valid": valid}) if return_parts else logits


# ─────────────────────────────── 预训练编码器版本 ───────────────────────────────


def fmt_context(s):
    return f"goal: {s.goal_text}\nstate: {s.obs_text}"


class NanoJevQwen(nn.Module):
    """NanoJev + Qwen3-0.6B：每个候选单独过一遍 Qwen，取最后一个 token 的隐藏状态作为 z_i。"""

    def __init__(self, name="Qwen/Qwen3-0.6B", train_last=4):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"
        self.lm = AutoModel.from_pretrained(name, dtype=torch.float32)
        for p in self.lm.parameters():
            p.requires_grad = False
        for blk in list(self.lm.layers[-train_last:]) + [self.lm.norm]:  # 只继续训练最后几层
            for p in blk.parameters():
                p.requires_grad = True
        dim = self.lm.config.hidden_size
        self.head = SetHead(dim, heads=8)

    def forward(self, states, return_parts=False):
        from env import candidates
        texts, owner = [], []
        for b, s in enumerate(states):
            for a in candidates(s):
                texts.append(f"{fmt_context(s)}\naction: {a}")
                owner.append(b)
        dev = next(self.head.parameters()).device
        enc = self.tok(texts, return_tensors="pt", padding=True).to(dev)
        h = self.lm(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        z = h[torch.arange(len(texts), device=dev), last]
        z, valid = group(z, torch.tensor(owner, device=dev), len(states))
        logits, attn = self.head(z, valid, return_attn=True)
        return (logits, {"z": z, "attn": attn, "valid": valid}) if return_parts else logits


class LayaBert(nn.Module):
    """Laya + ModernBERT-base：所有候选放进同一条输入，每个候选前放 [MASK]，取 [MASK] 位置的向量。"""

    def __init__(self, name="answerdotai/ModernBERT-base"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.enc = AutoModel.from_pretrained(name, dtype=torch.float32)
        self.head = SetHead(self.enc.config.hidden_size, set_attention=False)

    def forward(self, states, return_parts=False):
        from env import candidates
        m = self.tok.mask_token
        texts = [f"{fmt_context(s)}\nactions: " + " ".join(f"{m} {a}" for a in candidates(s)) for s in states]
        dev = next(self.head.parameters()).device
        enc = self.tok(texts, return_tensors="pt", padding=True).to(dev)
        h = self.enc(**enc).last_hidden_state
        is_mask = enc["input_ids"] == self.tok.mask_token_id
        K = int(is_mask.sum(1).max())
        z = h.new_zeros(len(states), K, h.shape[-1])
        valid = torch.zeros(len(states), K, dtype=torch.bool, device=dev)
        for b in range(len(states)):
            pos = is_mask[b].nonzero().squeeze(-1)
            z[b, :len(pos)] = h[b, pos]
            valid[b, :len(pos)] = True
        logits = self.head(z, valid)
        return (logits, {"z": z, "valid": valid}) if return_parts else logits


# ─────────────────────────────── 两种损失 ───────────────────────────────


def ce_loss(logits, p_star, valid):
    """交叉熵：−Σ p*·log p。最小值在 p = p* 处取到（等于 p* 的熵，不是 0）。"""
    logp = torch.log_softmax(logits, -1).masked_fill(~valid, 0.0)
    return -(p_star * logp).sum(-1).mean()


def brier_loss(logits, p_star, valid):
    """Brier：Σ (p − p*)²。p = p* 时等于 0。"""
    p = torch.softmax(logits, -1).masked_fill(~valid, 0.0)
    return ((p - p_star) ** 2).sum(-1).mean()


LOSSES = {"ce": ce_loss, "brier": brier_loss}
