"""
六个实验，每个对应视频里的一个论点（先运行 run_all.sh 训练好模型）：

    python experiments.py            # 全部实验，图片输出到 out/
    python experiments.py speed      # 只跑某一个：speed / explore / inside / loss / cost / generalize

    speed       GPT 逐 token 生成 vs Jev 一次给所有候选打分
    explore     锁住的文件：总选最高分会陷入循环，按概率采样才能试到解锁
    inside      一个具体状态：候选 → 向量 z → 候选间 attention → logit → softmax → 概率
    loss        交叉熵 vs Brier：都能学到目标概率吗？梯度有什么不同？
    cost        NanoJev（每个候选重复编码 obs+goal）vs Laya（一次编码）的计算量
    generalize  从零训练 vs 预训练编码器：换个说法、换个文件名还认得吗？
"""

import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import BOS, EOS, SEP, Vocab, batch_laya, batch_nanojev, context_ids, targets, words
from env import (FILES_TRAIN, PARAPHRASES, State, candidates, expert, random_episode_start, sample_states,
                 step)

HERE = Path(__file__).parent
OUT = HERE / "out"
CKPT = HERE / "checkpoints"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang HK", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_GPT, C_JEV, C_NANO, C_BERT, C_QWEN = "#888780", "#D85A30", "#3B8BD4", "#7F77DD", "#1D9E75"


# ─────────────────────────────── 工具 ───────────────────────────────


def detok(tokens: list) -> str:
    """把词级 token 拼回动作文本：open _ file report . md → open_file report.md"""
    text = ""
    for i, w in enumerate(tokens):
        glue = w in ("_", ".") or (i > 0 and tokens[i - 1] in ("_", "."))
        text += w if (glue or not text) else " " + w
    return text


@torch.no_grad()
def gpt_generate(model, vocab, s, greedy=True, rng=None, max_new=12, return_steps=False):
    """GPT 路线：一个 token 一个 token 地生成动作，每生成一个 token 都要完整跑一次模型。"""
    dev = next(model.parameters()).device
    ids = [BOS] + context_ids(vocab, s)[1:] + [SEP]
    out = []
    for _ in range(max_new):
        logits = model(torch.tensor([ids], device=dev))[0, -1]
        if greedy:
            t = int(logits.argmax())
        else:
            p = torch.softmax(logits, -1).cpu().tolist()
            t = (rng or random).choices(range(len(p)), weights=p)[0]
        if t == EOS:
            break
        ids.append(t)
        out.append(vocab.itos[t])
    text = detok(out)
    return (text, len(out) + 1) if return_steps else text


def load(name):
    """加载 checkpoints/<name>.pt，返回 (model, vocab, kind)。"""
    from models import GPTPolicy, Laya, LayaBert, NanoJev, NanoJevQwen
    ck = torch.load(CKPT / f"{name}.pt", map_location="cpu", weights_only=True)
    vocab = Vocab([])
    vocab.itos = ck["vocab"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    kind = ck["model"]
    if kind in ("gpt", "nanojev", "laya"):
        model = {"gpt": GPTPolicy, "nanojev": NanoJev, "laya": Laya}[kind](len(vocab.itos))
        model.load_state_dict(ck["state"])
    else:
        model = {"nanojev_qwen": NanoJevQwen, "laya_bert": LayaBert}[kind]()
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        assert not unexpected, unexpected
    return model.to(DEV).eval(), vocab, kind


@torch.no_grad()
def jev_probs(kind, model, vocab, states, parts=False):
    if kind == "nanojev":
        out = model(batch_nanojev(vocab, states, DEV), len(states), return_parts=parts)
    elif kind == "laya":
        out = model(batch_laya(vocab, states, DEV), len(states), return_parts=parts)
    else:
        out = model(states, return_parts=parts)
    logits, extra = out if parts else (out, None)
    p = torch.softmax(logits, -1)
    return (p, logits, extra) if parts else p


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → out/{name}")


def final_metrics(name):
    return json.loads((CKPT / f"{name}.json").read_text())


# ─────────────────────────────── 1. 速度 ───────────────────────────────


def exp_speed():
    print("[speed] GPT 逐 token 生成 vs Jev 一次打分")
    gpt, vocab, _ = load("gpt_ce")
    laya, _, _ = load("laya_ce")
    states = sample_states(150, seed=7)
    for s in states[:10]:                         # 预热，避免第一次调用的编译时间混进来
        gpt_generate(gpt, vocab, s); jev_probs("laya", laya, vocab, [s])
    sync()
    # 右图：真实任务里，每次决策的平均耗时
    rows = []
    for s in states:
        sync(); t0 = time.perf_counter()
        _, n_fwd = gpt_generate(gpt, vocab, s, return_steps=True); sync()
        t_gpt = time.perf_counter() - t0
        t0 = time.perf_counter()
        jev_probs("laya", laya, vocab, [s]); sync()
        rows.append((n_fwd, t_gpt * 1000, (time.perf_counter() - t0) * 1000, len(candidates(s))))
    r = np.array(rows)
    # 左图：强制 GPT 连续生成 n 个 token（不管 [EOS]），测出“每多一个 token 就多一次前向”
    ns = [1, 2, 4, 8, 12, 16, 24, 32]
    t_n = []
    with torch.no_grad():
        for n in ns:
            ts = []
            for s in states[:12]:
                ids = [BOS] + context_ids(vocab, s)[1:] + [SEP]
                sync(); t0 = time.perf_counter()
                for _ in range(n):
                    nxt = int(gpt(torch.tensor([ids], device=DEV))[0, -1].argmax())
                    ids.append(nxt)
                sync(); ts.append((time.perf_counter() - t0) * 1000)
            t_n.append(float(np.median(ts)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.plot(ns, t_n, "-o", color=C_GPT, label="GPT：生成 n 个 token（n 次前向，必须一个接一个）")
    ax.axhline(np.median(r[:, 2]), color=C_JEV, lw=2.5,
               label=f"Jev（Laya）：一次给全部 {r[:, 3].mean():.0f} 个左右的候选打分")
    ax.set_xlabel("动作描述的 token 数 n")
    ax.set_ylabel("耗时（毫秒）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("GPT 的耗时随动作长度线性增长；Jev 是一次前向", fontsize=11)
    ax = axes[1]
    labels = ["GPT\n逐 token 生成", "Jev（Laya）\n一次打分"]
    vals = [r[:, 1].mean(), r[:, 2].mean()]
    bars = ax.bar(labels, vals, color=[C_GPT, C_JEV], width=0.55)
    for b, v, k in zip(bars, vals, [f"平均 {r[:, 0].mean():.1f} 次前向", "1 次前向"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} ms\n{k}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("平均耗时（毫秒）")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title(f"真实任务：150 个状态的平均（动作只有 3–6 个 token）", fontsize=11)
    fig.suptitle("实验 1 · 速度：GPT 要一个 token 一个 token 地生成，Jev 一次前向就够", fontsize=13)
    fig.tight_layout()
    save(fig, "1_speed.png")
    return {"gpt_ms": float(r[:, 1].mean()), "jev_ms": float(r[:, 2].mean()), "gpt_forwards": float(r[:, 0].mean()),
            "gpt_ms_by_n": dict(zip(ns, t_n))}


# ─────────────────────────────── 2. 探索 ───────────────────────────────


def run_episode(policy, s, rng, max_steps=12):
    trace = []
    for _ in range(max_steps):
        a = policy(s, rng)
        r = step(s, a)
        trace.append((a, r))
        if r in ("success", "disaster"):
            return r == "success", trace
    return False, trace


def exp_explore():
    print("[explore] 锁住的文件：greedy vs 按概率采样")
    laya, vocab, _ = load("laya_ce")

    def jev_policy(greedy):
        def pol(s, rng):
            p = jev_probs("laya", laya, vocab, [s])[0][:len(candidates(s))].cpu().tolist()
            c = candidates(s)
            return c[int(np.argmax(p))] if greedy else rng.choices(c, weights=p)[0]
        return pol

    res = {}
    for locked in (False, True):
        for name, pol in [("greedy", jev_policy(True)), ("sample", jev_policy(False))]:
            rng = random.Random(1)
            wins = []
            for _ in range(200):
                s = random_episode_start(rng, p_locked=1.0 if locked else 0.0)
                if s.verb == "delete":            # 删除不需要打开文件，锁不锁都一样，只看编辑/阅读
                    s.verb = rng.choice(["edit", "read"])
                ok, _ = run_episode(pol, s, rng)
                wins.append(ok)
            res[(locked, name)] = float(np.mean(wins))
    # 一个具体的循环例子
    rng = random.Random(3)
    s = State(files=["notes.txt", "report.md", "todo.md"], target="report.md", verb="edit", locked={"report.md"})
    _, greedy_trace = run_episode(jev_policy(True), s, rng, max_steps=7)
    for seed in range(100):                     # 示例：第一条成功的采样轨迹（成功率看左图的统计）
        s = State(files=["notes.txt", "report.md", "todo.md"], target="report.md", verb="edit", locked={"report.md"})
        ok, sample_trace = run_episode(jev_policy(False), s, random.Random(seed), max_steps=12)
        if ok:
            break

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    x = np.arange(2)
    g = [res[(False, "greedy")], res[(True, "greedy")]]
    sm = [res[(False, "sample")], res[(True, "sample")]]
    ax.bar(x - 0.18, g, 0.36, color=C_GPT, label="总选最高分（greedy）")
    ax.bar(x + 0.18, sm, 0.36, color=C_JEV, label="按概率采样")
    for i in range(2):
        ax.text(i - 0.18, g[i] + 0.02, f"{g[i]:.0%}", ha="center")
        ax.text(i + 0.18, sm[i] + 0.02, f"{sm[i]:.0%}", ha="center")
    ax.set_xticks(x, ["文件没锁", "文件被锁（观测里看不出来）"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("任务成功率（编辑 / 阅读，最多 12 步）")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("同一个模型，只是选动作的方式不同", fontsize=11)
    ax = axes[1]
    ax.axis("off")
    lines = ["目标：edit report.md（report.md 被锁住）", "", "总选最高分："]
    lines += [f"  {i + 1}. {a}  →  {r}" for i, (a, r) in enumerate(greedy_trace)]
    lines += ["  …… 一直在原地打转", "", "按概率采样（一次采样的例子）："]
    lines += [f"  {i + 1}. {a}  →  {r}" for i, (a, r) in enumerate(sample_trace)]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=10.5, linespacing=1.45)
    fig.suptitle("实验 2 · 探索：只选分数最高的动作，出错后会陷入循环；有了概率才能尝试别的动作", fontsize=13)
    fig.tight_layout()
    save(fig, "2_explore.png")
    return {f"{'locked' if k[0] else 'unlocked'}_{k[1]}": v for k, v in res.items()}


# ─────────────────────────────── 3. 模型内部 ───────────────────────────────


def demo_state():
    return State(files=["notes.txt", "report.md", "todo.md"], target="report.md", verb="edit",
                 known=["notes.txt", "report.md", "todo.md"])


def exp_inside():
    print("[inside] 一个状态从候选到概率的全过程（NanoJev）")
    nano, vocab, _ = load("nanojev_ce")
    s = demo_state()
    cands = candidates(s)
    p, logits, parts = jev_probs("nanojev", nano, vocab, [s], parts=True)
    K = len(cands)
    z = parts["z"][0, :K].cpu().numpy()
    attn = parts["attn"][0, :K, :K].cpu().numpy()
    lg = logits[0, :K].cpu().numpy()
    pr = p[0, :K].cpu().numpy()
    ps = np.array(list(expert(s).values()))

    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 0.8, 1.2])
    ax = fig.add_subplot(gs[0])
    ax.imshow(z[:, :24], cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(K), cands, fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel("向量的前 24 维（共 128 维）")
    ax.set_title("① f(obs, goal, action) → 向量 z", fontsize=11)
    ax = fig.add_subplot(gs[1])
    ax.imshow(attn, cmap="Oranges", aspect="auto")
    ax.set_yticks(range(K), [""] * K)
    ax.set_xticks(range(K), [str(i + 1) for i in range(K)], fontsize=8)
    ax.set_xlabel("看向哪个候选（按左边顺序编号）")
    ax.set_title("② 候选之间的 attention", fontsize=11)
    ax = fig.add_subplot(gs[2])
    ax.barh(range(K), lg, color=[C_JEV if v == lg.max() else C_GPT for v in lg])
    ax.invert_yaxis()
    ax.set_yticks(range(K), [""] * K)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_title("③ Wx + b → logit（实数）", fontsize=11)
    ax = fig.add_subplot(gs[3])
    y = np.arange(K)
    ax.barh(y - 0.2, pr, 0.4, color=C_JEV, label="模型预测 p")
    ax.barh(y + 0.2, ps, 0.4, color=C_GPT, label="目标 p*（专家）")
    for i in range(K):
        if max(pr[i], ps[i]) > 0.02:
            ax.text(max(pr[i], ps[i]) + 0.02, i, f"{pr[i]:.2f} / {ps[i]:.2f}", va="center", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_yticks(y, [""] * K)
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("④ softmax → 概率，和目标对比", fontsize=11)
    fig.suptitle(f"实验 3 · 模型内部：目标 “{s.goal_text}”，观测 “{s.obs_text}”，共 {K} 个候选", fontsize=13)
    fig.tight_layout()
    save(fig, "3_inside.png")
    return {"candidates": cands, "p": pr.round(3).tolist(), "p_star": ps.round(3).tolist(), "logits": lg.round(2).tolist()}


# ─────────────────────────────── 4. 交叉熵 vs Brier ───────────────────────────────


def exp_loss():
    print("[loss] 交叉熵 vs Brier")
    test = sample_states(600, seed=101)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    out = {}
    for name, color, label in [("laya_ce", C_JEV, "交叉熵 −Σ p*·log p"), ("laya_brier", C_NANO, "Brier Σ (p − p*)²")]:
        h = final_metrics(name)["history"]
        axes[0].plot([r["step"] for r in h], [r["test"]["kl"] for r in h], "-o", ms=3, color=color, label=label)
        model, vocab, kind = load(name)
        ps, pt = [], []
        for i in range(0, len(test), 64):
            chunk = test[i:i + 64]
            p_star, valid = targets(chunk, DEV)
            p = jev_probs(kind, model, vocab, chunk)
            ps.append(p[valid].cpu().numpy())
            pt.append(p_star[valid].cpu().numpy())
        ps, pt = np.concatenate(ps), np.concatenate(pt)
        ax = axes[1] if name == "laya_ce" else axes[2]
        jitter = np.random.default_rng(0).normal(0, 0.008, len(pt))
        ax.scatter(pt + jitter, ps, s=6, alpha=0.25, color=color)
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
        for v in sorted(set(np.round(pt, 2))):
            sel = np.round(pt, 2) == v
            ax.plot(v, ps[sel].mean(), "D", color="black", ms=6)
        ax.set_xlabel("目标概率 p*")
        ax.set_ylabel("模型预测概率 p")
        ax.set_title(f"{label}\n黑色菱形 = 每个目标值下预测的平均值", fontsize=10.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        out[name] = final_metrics(name)["final"]["test"]
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("测试集 KL(p* ‖ p)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("两种损失都在逼近目标分布", fontsize=11)
    fig.suptitle("实验 4 · 训练目标：交叉熵和 Brier 都能学到目标概率（点越贴近对角线越准）", fontsize=13)
    fig.tight_layout()
    save(fig, "4_loss.png")

    # 两种损失的梯度：二分类、目标 p* = 0.7，对 logit z 求导
    fig, ax = plt.subplots(figsize=(6.5, 4))
    p = np.linspace(0.001, 0.999, 400)
    ax.plot(p, p - 0.7, color=C_JEV, label="交叉熵：∂L/∂z = p − p*")
    ax.plot(p, 4 * (p - 0.7) * p * (1 - p), color=C_NANO, label="Brier：∂L/∂z = 4(p − p*)·p(1 − p)")
    ax.axhline(0, color="#555", lw=0.8)
    ax.axvline(0.7, color="#555", lw=0.8, ls="--")
    ax.text(0.71, 0.25, "p = p* = 0.7 时\n两者梯度都为 0", fontsize=9)
    ax.set_xlabel("当前预测概率 p")
    ax.set_ylabel("对 logit 的梯度")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("梯度对比（二分类，目标 p* = 0.7）：\n预测非常错（p → 0）时，Brier 的梯度会变得很小", fontsize=10.5)
    fig.tight_layout()
    save(fig, "4b_gradient.png")
    return out


# ─────────────────────────────── 5. NanoJev vs Laya 的计算量 ───────────────────────────────


def exp_cost():
    print("[cost] NanoJev vs Laya 的计算量")
    nano, vocab, _ = load("nanojev_ce")
    laya, _, _ = load("laya_ce")
    states = sample_states(400, seed=9)
    rows = []
    for s in states:
        b1 = batch_nanojev(vocab, [s], DEV)
        b2 = batch_laya(vocab, [s], DEV)
        rows.append((len(candidates(s)), int((b1["ids"] != 0).sum()), int((b2["ids"] != 0).sum())))
    r = np.array(rows)
    Ks = sorted(set(r[:, 0]))
    t_nano, t_laya, Ks_t = [], [], [k for k in Ks if (r[:, 0] == k).sum() >= 3]
    for k in Ks_t:
        group = [s for s in states if len(candidates(s)) == k][:3]
        group = (group * 11)[:32]                 # 一次批量算 32 个状态，摊薄 GPU 调用开销，测到真实计算量
        for model, kind, acc in [(nano, "nanojev", t_nano), (laya, "laya", t_laya)]:
            jev_probs(kind, model, vocab, group); sync()
            t0 = time.perf_counter()
            for _ in range(10):
                jev_probs(kind, model, vocab, group)
            sync()
            acc.append((time.perf_counter() - t0) / 10 / len(group) * 1000)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    ax.plot(Ks, [r[r[:, 0] == k, 1].mean() for k in Ks], "-o", color=C_NANO, label="NanoJev：每个候选都带上 obs + goal")
    ax.plot(Ks, [r[r[:, 0] == k, 2].mean() for k in Ks], "-o", color=C_JEV, label="Laya：obs + goal 只出现一次")
    ax.set_xlabel("候选动作数量 K")
    ax.set_ylabel("每次决策要处理的 token 数")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("NanoJev 把 obs 和 goal 重复编码了 K 次", fontsize=11)
    ax = axes[1]
    ax.plot(Ks_t, t_nano, "-o", color=C_NANO, label="NanoJev")
    ax.plot(Ks_t, t_laya, "-o", color=C_JEV, label="Laya")
    ax.set_xlabel("候选动作数量 K")
    ax.set_ylabel("每个状态的耗时（毫秒，一批 32 个状态）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("实测耗时：候选越多，NanoJev 越慢", fontsize=11)
    fig.suptitle("实验 5 · NanoJev vs Laya：候选越多，重复计算越多", fontsize=13)
    fig.tight_layout()
    save(fig, "5_cost.png")
    k_max = Ks[-1]
    return {"K": int(k_max), "tokens_nanojev": float(r[r[:, 0] == k_max, 1].mean()),
            "tokens_laya": float(r[r[:, 0] == k_max, 2].mean())}


# ─────────────────────────────── 6. 预训练 vs 从零训练 ───────────────────────────────


def exp_generalize():
    print("[generalize] 从零训练 vs 预训练编码器")
    names = [("gpt_ce", "GPT 逐 token 生成（从零训练）", C_GPT),
             ("laya_ce", "Laya（从零训练）", C_JEV), ("nanojev_ce", "NanoJev（从零训练）", C_NANO),
             ("laya_bert_ce", "Laya + ModernBERT", C_BERT), ("nanojev_qwen_ce", "NanoJev + Qwen3-0.6B", C_QWEN)]
    splits = [("test", "同分布测试"), ("paraphrase", "动词换成没见过的同义词\n（modify / view / remove …）"),
              ("unseen_files", "文件名都没见过\n（invoice.pdf / resume.docx …）")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(splits))
    w = 0.16
    table = {}
    for i, (n, label, c) in enumerate(names):
        f = final_metrics(n)["final"]
        vals = [f[sp]["top1"] for sp, _ in splits]
        table[n] = f
        bars = ax.bar(x + (i - 2) * w, vals, w, color=c, label=label, hatch="//" if n == "gpt_ce" else None,
                      edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.5)
    inv = table["gpt_ce"]["unseen_files"]["invalid"]
    ax.annotate(f"GPT 有 {inv:.0%} 的时候\n生成了根本不存在的动作", xy=(2 - 2 * w, table["gpt_ce"]["unseen_files"]["top1"]),
                xytext=(1.62, 1.13), fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#555"),
                bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    ax.set_xticks(x, [t for _, t in splits], fontsize=9.5)
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("top-1：最高概率的动作 = 专家最想做的动作")
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.set_title("没见过的文件名问题不大；换个说法，只有 Qwen3 还能稳定理解", fontsize=11)

    # 具体例子：已经打开文件，目标用了没见过的动词
    ax = axes[1]
    s = State(files=["report.md", "notes.txt"], target="report.md", verb="edit", known=["report.md", "notes.txt"],
              opened="report.md", goal_word="modify")
    cands = candidates(s)
    probs = {}
    pair = [x for x in (names[1], names[4]) if (CKPT / f"{x[0]}.pt").exists()]
    if len(pair) < 2:
        print("  （没有找到 nanojev_qwen_ce.pt：右边的例子只画从零训练的模型；运行 ./run_all.sh 可以生成）")
    for n, label, c in pair:
        model, vocab, kind = load(n)
        probs[n] = jev_probs(kind, model, vocab, [s])[0, :len(cands)].cpu().numpy()
        del model
    top = set()
    for n in probs:                                   # 两个模型各自最看好的动作都列出来
        top |= set(np.argsort(-probs[n])[:3].tolist())
    idx = sorted(top, key=lambda i: -max(probs[n][i] for n in probs))
    show = [cands[i] for i in idx]
    y = np.arange(len(show))
    for j, (n, label, c) in enumerate(pair):
        ax.barh(y + (j - 0.5) * 0.36, probs[n][idx], 0.36, color=c, label=label)
        for k, v in zip(y, probs[n][idx]):
            ax.text(v + 0.02, k + (j - 0.5) * 0.36, f"{v:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y, show, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title(f"例子：目标 “{s.goal_text}”，report.md 已打开\n训练时只见过 edit，没见过 modify（专家答案：edit_file）", fontsize=10.5)
    fig.suptitle("实验 6 · 为什么要复用预训练模型：表征不用从头学，没见过的说法也能理解", fontsize=13)
    fig.tight_layout()
    save(fig, "6_generalize.png")
    res = {n: {sp: round(table[n][sp]["top1"], 3) for sp, _ in splits} for n, _, _ in names}
    res["gpt_invalid_unseen"] = inv
    return res


EXPERIMENTS = {"speed": exp_speed, "explore": exp_explore, "inside": exp_inside, "loss": exp_loss,
               "cost": exp_cost, "generalize": exp_generalize}

if __name__ == "__main__":
    which = sys.argv[1:] or list(EXPERIMENTS)
    results = {}
    summary = OUT / "results.json"
    if summary.exists():
        results = json.loads(summary.read_text())
    for k in which:
        results[k] = EXPERIMENTS[k]()
    OUT.mkdir(exist_ok=True)
    summary.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("完成，结果汇总在 out/results.json")
