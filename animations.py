"""
把几个实验做成动图（给文章 / README 用），输出到 out/gifs/。

    python animations.py            # 全部
    python animations.py speed      # 只做某一个：speed / explore / train / pack

    speed    GPT 一个 token 一个 token 地写出动作，Jev 一次给所有候选打分（按实测耗时播放）
    explore  文件被锁住：greedy 原地打转，按概率采样试到了解锁（真实运行的轨迹）
    train    从零训练时，一个状态上的概率怎么一步步靠近专家目标 p*（交叉熵 vs Brier）
    pack     候选越来越多时，NanoJev 和 Laya 各要处理多少个 token
"""

import json
import math
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from data import Vocab, batch_laya, batch_nanojev, targets
from env import State, candidates, expert, sample_states
from experiments import C_GPT, C_JEV, C_NANO, DEV, OUT, demo_state, jev_probs, load, run_episode
from models import LOSSES, Laya

GIFS = OUT / "gifs"
FPS = 10


def save_gif(anim, name, fps=FPS):
    GIFS.mkdir(parents=True, exist_ok=True)
    anim.save(GIFS / name, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(anim._fig)
    print(f"  → out/gifs/{name}")


# ─────────────────────────────── 1. 速度 ───────────────────────────────


def anim_speed():
    """时间轴按实测耗时：GPT 每个 token 一次前向（约 1.7 ms），Laya 一次前向给全部候选打分。"""
    print("[speed]")
    res = json.loads((OUT / "results.json").read_text())["speed"]
    per_tok = res["gpt_ms_by_n"]["32"] / 32
    jev_ms = res["jev_ms"]
    laya, vocab, _ = load("laya_ce")
    s = demo_state()
    cands = candidates(s)
    p = jev_probs("laya", laya, vocab, [s])[0][:len(cands)].cpu().numpy()
    toks = ["open", "_", "file", " report", ".", "md", "[EOS]"]   # GPT 要写出的动作（每个都是一次前向）
    total = per_tok * len(toks)
    T = np.arange(0, total + 3.0, 0.25)                           # 1 帧 = 0.25 ms，按真实比例慢放

    fig = plt.figure(figsize=(12, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.18], hspace=0.35, wspace=0.25)
    ax_g, ax_j, ax_t = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    fig.suptitle(f"同一个状态、同样 0.7M 参数：GPT 生成一个动作 vs Jev 给 {len(cands)} 个候选打分（按实测耗时慢放）",
                 fontsize=12.5)

    def draw(t):
        for ax in (ax_g, ax_j, ax_t):
            ax.clear()
        # GPT：逐 token 写
        ax_g.axis("off")
        n = min(len(toks), int(t // per_tok))
        ax_g.set_title("GPT：一个 token 一个 token 地写", fontsize=12, color=C_GPT, loc="left")
        ax_g.text(0, 0.85, "目标 edit report.md\n观测 known notes.txt report.md todo.md", fontsize=10, color="#555",
                  transform=ax_g.transAxes, va="top")
        text = "".join(toks[:n]).replace("[EOS]", "")
        cursor = "▌" if n < len(toks) else ""
        ax_g.text(0, 0.5, "输出", fontsize=12, color="#555", transform=ax_g.transAxes)
        ax_g.text(0.1, 0.5, text + cursor, fontsize=17, family="monospace", transform=ax_g.transAxes)
        ax_g.text(0, 0.28, f"已经跑了 {n} 次前向", fontsize=11, transform=ax_g.transAxes)
        if n == len(toks):
            ax_g.text(0, 0.1, f"完成：{total:.1f} ms，只得到 1 个动作", fontsize=12, color=C_GPT, fontweight="bold",
                      transform=ax_g.transAxes)
        # Jev：一次前向后所有候选同时有概率
        done = t >= jev_ms
        y = np.arange(len(cands))
        ax_j.barh(y, p if done else np.zeros_like(p), color=[C_JEV if v == p.max() else "#F0B59B" for v in p])
        ax_j.set_yticks(y, cands, fontsize=8.5)
        ax_j.invert_yaxis()
        ax_j.set_xlim(0, 1)
        ax_j.set_xlabel("概率")
        title = f"Jev：1 次前向，{len(cands)} 个候选同时打分" + (f"（{jev_ms:.1f} ms 完成）" if done else "")
        ax_j.set_title(title, fontsize=12, color=C_JEV, loc="left")
        # 时间轴
        ax_t.set_xlim(0, total + 1)
        ax_t.set_ylim(0, 2)
        ax_t.barh(1.4, min(t, total), 0.5, color=C_GPT)
        ax_t.barh(0.5, min(t, jev_ms), 0.5, color=C_JEV)
        for k in range(1, len(toks)):
            ax_t.axvline(k * per_tok, ymin=0.5, ymax=0.95, color="white", lw=1.5)
        ax_t.set_yticks([0.5, 1.4], ["Jev", "GPT"])
        ax_t.set_xlabel(f"时间（毫秒）　t = {min(t, total):.1f} ms")
        for sp in ("top", "right"):
            ax_t.spines[sp].set_visible(False)

    anim = FuncAnimation(fig, lambda i: draw(T[i]), frames=len(T))
    save_gif(anim, "speed.gif")


# ─────────────────────────────── 2. 探索 ───────────────────────────────


def anim_explore():
    print("[explore]")
    laya, vocab, _ = load("laya_ce")

    def pol(greedy):
        def f(s, rng):
            c = candidates(s)
            p = jev_probs("laya", laya, vocab, [s])[0][:len(c)].cpu().tolist()
            a = c[int(np.argmax(p))] if greedy else rng.choices(c, weights=p)[0]
            f.last_p = p[c.index(a)]
            return a
        return f

    def start():
        return State(files=["notes.txt", "report.md", "todo.md"], target="report.md", verb="edit", locked={"report.md"})

    def traced(policy, rng, max_steps):
        """和 run_episode 一样，额外记下每一步选中动作的概率。"""
        s, out = start(), []
        from env import step
        for _ in range(max_steps):
            a = policy(s, rng)
            r = step(s, a)
            out.append((a, policy.last_p, r))
            if r in ("success", "disaster"):
                break
        return out

    greedy = traced(pol(True), random.Random(0), 8)
    for seed in range(100):                       # 第一条成功的采样轨迹（和实验 2 的示例一致）
        sample = traced(pol(False), random.Random(seed), 12)
        if sample[-1][2] == "success":
            break
    n = max(len(greedy), len(sample))
    zh = {"ok": "成功", "fail": "失败（被锁）", "success": "任务完成 ✓", "disaster": "删错了"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    fig.suptitle("目标 edit report.md，report.md 被锁住（agent 看不到）。同一个模型，只是选动作的方式不同", fontsize=12.5)

    def draw(i):
        k = min(n, i // 6 + 1)                    # 每 6 帧多走一步
        for ax, trace, title, col in [(axes[0], greedy, "总选概率最高的（greedy）", C_GPT),
                                      (axes[1], sample, "按概率采样", C_JEV)]:
            ax.clear(); ax.axis("off")
            ax.set_title(title, fontsize=13, color=col, loc="left", fontweight="bold")
            for j, (a, p, r) in enumerate(trace[:k]):
                c = "#C53030" if r in ("fail", "disaster") else ("#1D9E75" if r == "success" else "#333")
                hl = a.startswith("unlock")
                ax.text(0, 0.92 - j * 0.085, f"{j + 1:>2}. {a:<24} p={p:.2f}  → {zh[r]}", fontsize=11,
                        family="Arial Unicode MS", color=c, fontweight="bold" if hl else "normal",
                        transform=ax.transAxes)
            if trace is greedy and k >= len(greedy):
                ax.text(0, 0.92 - len(greedy) * 0.085, "    …… 状态不变，选择也永远不变", fontsize=11, color=C_GPT,
                        transform=ax.transAxes)
            if trace is sample and k >= len(sample):
                ax.text(0, 0.92 - len(sample) * 0.085, "    unlock 只有 15% 的概率，但迟早会被试到", fontsize=11,
                        color=C_JEV, transform=ax.transAxes)

    anim = FuncAnimation(fig, draw, frames=n * 6 + 25)
    save_gif(anim, "explore.gif")


# ─────────────────────────────── 3. 训练过程 ───────────────────────────────


def train_snapshots(loss_name, s, steps=6000, every=150):
    """和 train.py 相同的设置从零训练 Laya，每隔一段记录一次 s 上的概率。"""
    from train import datasets
    torch.manual_seed(0)
    rng = random.Random(0)
    data = datasets()["train"]
    vocab = Vocab(data)
    model = Laya(len(vocab)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    snaps = []

    @torch.no_grad()
    def probe(step):
        model.eval()
        p = torch.softmax(model(batch_laya(vocab, [s], DEV), 1)[0], -1)[:len(candidates(s))].cpu().numpy()
        model.train()
        snaps.append((step, p))

    probe(0)
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = 1e-3 * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        states = rng.sample(data, 64)
        p_star, valid = targets(states, DEV)
        loss = LOSSES[loss_name](model(batch_laya(vocab, states, DEV), 64), p_star, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % every == 0:
            probe(step)
    print(f"  {loss_name}: {len(snaps)} 个快照")
    return snaps


def anim_train():
    print("[train]  从零训练两个 Laya（交叉熵 / Brier），各约 3 分钟")
    s = demo_state()
    cands = candidates(s)
    ps = np.array(list(expert(s).values()))
    cache = GIFS / "train_snapshots.npz"          # 训练一次要几分钟，结果缓存起来，改图时不用重训
    if cache.exists():
        z = np.load(cache)
        runs = {k: list(zip(z["steps"].tolist(), z[k])) for k in ("ce", "brier")}
    else:
        runs = {"ce": train_snapshots("ce", s), "brier": train_snapshots("brier", s)}
        GIFS.mkdir(parents=True, exist_ok=True)
        np.savez(cache, steps=[st for st, _ in runs["ce"]], **{k: np.stack([p for _, p in v]) for k, v in runs.items()})
    y = np.arange(len(cands))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    fig.suptitle("训练过程：同一个状态上，模型的概率怎么靠近专家目标 p*（灰框）", fontsize=13, y=0.99)
    fig.subplots_adjust(top=0.82, wspace=0.12)

    def draw(i):
        i = min(i, len(runs["ce"]) - 1)
        for ax, key, title in [(axes[0], "ce", "交叉熵  −Σ p* log p"), (axes[1], "brier", "Brier  Σ (p − p*)²")]:
            ax.clear()
            step, p = runs[key][i]
            ax.barh(y, ps, 0.8, fill=False, edgecolor="#888", lw=1.5, label="专家目标 p*")
            ax.barh(y, p, 0.55, color=C_JEV if key == "ce" else C_NANO, label="模型概率 p")
            kl = float(np.sum(ps * (np.log(np.clip(ps, 1e-9, 1)) - np.log(np.clip(p, 1e-9, 1)))))
            ax.set_title(f"{title}\n第 {step} 步　KL(p*‖p) = {kl:.3f}", fontsize=11.5, loc="left")
            ax.set_xlim(0, 1)
            ax.set_yticks(y, cands, fontsize=9)
            ax.legend(loc="lower right", fontsize=9)
        axes[0].invert_yaxis()

    anim = FuncAnimation(fig, draw, frames=len(runs["ce"]) + 15)
    save_gif(anim, "train.gif")


# ─────────────────────────────── 4. NanoJev vs Laya ───────────────────────────────


def anim_pack():
    print("[pack]")
    states = sample_states(3000, seed=5)
    vocab = Vocab(states)
    s = max(states, key=lambda x: len(candidates(x)))           # 候选最多的状态
    cands = candidates(s)
    K = len(cands)
    nano, laya = [], []
    for k in range(1, K + 1):                                     # 只保留前 k 个候选时各要多少 token
        nano.append(sum(len(x) for x in _nano_seqs(vocab, s, k)))
        laya.append(len(_laya_seq(vocab, s, k)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.suptitle("候选越来越多：NanoJev 每个候选都要重复一遍 goal + obs，Laya 一条序列装下全部", fontsize=12.5)

    def draw(i):
        k = min(K, i + 1)
        ax = axes[0]; ax.clear(); ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("每个方块 = 一个 token　（蓝 goal　绿 obs　橙 [MASK]　灰 候选）", fontsize=10.5, loc="left")
        ax.text(0, 0.96, f"NanoJev：{k} 条序列", fontsize=11, color=C_NANO, fontweight="bold")
        rows = _nano_seqs(vocab, s, k, colored=True)
        h = 0.52 / (1.15 * K)
        for r, seq in enumerate(rows):
            for j, c in enumerate(seq):
                ax.add_patch(plt.Rectangle((j * 0.016, 0.92 - (r + 1) * h * 1.15), 0.014, h * 0.75, color=c))
        ax.text(0, 0.25, f"Laya：1 条序列", fontsize=11, color=C_JEV, fontweight="bold")
        seq = _laya_seq(vocab, s, k, colored=True)
        per_row = 60
        for j, c in enumerate(seq):
            ax.add_patch(plt.Rectangle(((j % per_row) * 0.016, 0.18 - (j // per_row) * 0.045), 0.014, 0.035, color=c))
        ax = axes[1]; ax.clear()
        xs = np.arange(1, k + 1)
        ax.plot(xs, nano[:k], "-o", ms=3, color=C_NANO, label=f"NanoJev：{nano[k - 1]} 个 token")
        ax.plot(xs, laya[:k], "-o", ms=3, color=C_JEV, label=f"Laya：{laya[k - 1]} 个 token")
        ax.set_xlim(0, K + 1); ax.set_ylim(0, max(nano) * 1.1)
        ax.set_xlabel("候选数 K"); ax.set_ylabel("一次决策要处理的 token 数")
        ax.legend(loc="upper left", fontsize=10); ax.grid(alpha=0.3)

    anim = FuncAnimation(fig, draw, frames=K + 15)
    save_gif(anim, "pack.gif", fps=4)


C_GOAL, C_OBS, C_MASKC, C_ACT, C_SPEC = "#3B8BD4", "#1D9E75", "#D85A30", "#B4B2A9", "#555555"


def _ctx(vocab, s, colored):
    from data import CLS, SEP
    g, o = vocab.encode("goal " + s.goal_text), vocab.encode(s.obs_text)
    if colored:
        return [C_SPEC] + [C_GOAL] * len(g) + [C_SPEC] + [C_OBS] * len(o)
    return [CLS] + g + [SEP] + o


def _nano_seqs(vocab, s, k, colored=False):
    """和 data.batch_nanojev 相同的打包方式，只取前 k 个候选。"""
    from data import SEP
    out = []
    for a in candidates(s)[:k]:
        act = vocab.encode(a)
        out.append(_ctx(vocab, s, colored) + ([C_SPEC] + [C_ACT] * len(act) if colored else [SEP] + act))
    return out


def _laya_seq(vocab, s, k, colored=False):
    """和 data.batch_laya 相同的打包方式，只取前 k 个候选。"""
    from data import MASK
    ids = _ctx(vocab, s, colored)
    for a in candidates(s)[:k]:
        act = vocab.encode(a)
        ids = ids + ([C_MASKC] + [C_ACT] * len(act) if colored else [MASK] + act)
    return ids


if __name__ == "__main__":
    jobs = {"speed": anim_speed, "explore": anim_explore, "train": anim_train, "pack": anim_pack}
    for name in sys.argv[1:] or jobs:
        jobs[name]()
