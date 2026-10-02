"""
一个具体场景，从输入、推理到输出，一步一步可视化（用从零训练的 Laya 模型）。

    python scenario.py            # 输出 out/scenario/step_*.png、out/scenario.gif、out/7_scenario.png

场景：目标 “edit report.md”，report.md 被锁住了（agent 看不到），一开始什么文件都不知道。
每一步：
    ① 输入：目标 + 当前观测
    ② Laya 的输入序列：[CLS] goal [SEP] obs [MASK] 候选1 [MASK] 候选2 …
    ③ 推理：最后一层里，每个候选的 [MASK] 在看输入里的哪些词
    ④ 每个 [MASK] 的向量 → logit → softmax 概率（对照专家的目标概率）
    ⑤ 输出：按概率采样一个动作 → 环境反馈 → 状态更新
"""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from data import MASK, PAD, batch_laya
from env import State, candidates, copy_state, expert, step
from experiments import C_GPT, C_JEV, DEV, OUT, load

C_GOAL, C_OBS, C_MASKC = "#3B8BD4", "#1D9E75", "#D85A30"
RESULT_ZH = {"ok": "执行成功", "fail": "失败：文件被锁住，状态没变", "success": "任务完成 ✓",
             "disaster": "删错文件，任务失败"}


@torch.no_grad()
def laya_with_attention(model, vocab, s):
    """跑一遍 Laya，同时取出最后一层的 attention（每个 [MASK] 看向每个 token 的权重）。"""
    b = batch_laya(vocab, [s], DEV)
    last = model.enc.layers.layers[-1]
    cache = {}
    hook = last.register_forward_pre_hook(lambda mod, args, kwargs: cache.update(x=args[0]), with_kwargs=True)
    logits = model(b, 1)
    hook.remove()
    x = last.norm1(cache["x"])
    _, attn = last.self_attn(x, x, x, key_padding_mask=b["ids"] == PAD, need_weights=True, average_attn_weights=True)
    ids = b["ids"][0].tolist()
    pos = b["mask_pos"][0].tolist()
    K = len(candidates(s))
    return ids, pos[:K], attn[0, pos[:K]].cpu().numpy(), logits[0, :K].cpu().numpy()


def draw_tokens(ax, toks, colors, y0=0.9, x0=0.0, width=1.0, fs=10):
    """把 token 画成一排排小方块（自动换行），返回每个 token 的位置。"""
    x, y, h = x0, y0, 0.095
    for t, c in zip(toks, colors):
        w = 0.011 * max(len(t), 1) + 0.016
        if x + w > x0 + width:
            x, y = x0, y - h * 1.25
        ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.002,rounding_size=0.01",
                                    fc=c, ec="white", lw=1, transform=ax.transAxes))
        ax.text(x + w / 2, y - h / 2, t, ha="center", va="center", fontsize=fs, transform=ax.transAxes,
                color="white" if c not in ("#EEEEEE", "#F2F2F2") else "#333")
        x += w + 0.004
    return y - h


def token_colors(toks):
    """goal 蓝色、obs 绿色、[MASK] 橙色、候选动作灰色。"""
    cols, part = [], "cls"
    for t in toks:
        if t == "[CLS]":
            cols.append("#555555"); continue
        if t == "goal" and part == "cls":
            part = "goal"
        if t == "[SEP]":
            part = "obs"; cols.append("#555555"); continue
        if t == "[MASK]":
            part = "act"; cols.append(C_MASKC); continue
        cols.append({"goal": C_GOAL, "obs": C_OBS, "act": "#9C9A92"}.get(part, "#555555"))
    return cols


def draw_step(k, s, model, vocab, action, result, after, out_path):
    cands = candidates(s)
    ids, pos, attn, logits = laya_with_attention(model, vocab, s)
    p = np.exp(logits - logits.max()); p /= p.sum()
    ps = np.array(list(expert(s).values()))
    toks = [vocab.itos[i] for i in ids]
    K = len(cands)
    chosen = cands.index(action)

    fig = plt.figure(figsize=(16, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 1.25, 0.55], width_ratios=[1.35, 1], hspace=0.38, wspace=0.12)
    # ① 输入
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title(f"第 {k} 步 · ① 输入", loc="left", fontsize=13, fontweight="bold")
    ax.text(0, 0.72, "目标 goal", color=C_GOAL, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.15, 0.72, s.goal_text, fontsize=13, transform=ax.transAxes)
    ax.text(0, 0.38, "观测 obs", color=C_OBS, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.15, 0.38, s.obs_text, fontsize=13, transform=ax.transAxes)
    ax.text(0, 0.04, f"候选 {K} 个（由任务 + 当前状态映射出来）：" + "、".join(cands), fontsize=9.5, color="#555",
            transform=ax.transAxes, wrap=True)
    # ② Laya 的输入序列
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("② Laya 的输入：所有候选放进同一条序列，每个前面放一个 [MASK]", loc="left", fontsize=11)
    draw_tokens(ax, toks, token_colors(toks), y0=0.95, fs=8)
    # ③ 推理：[MASK] 的 attention
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(attn, cmap="Oranges", aspect="auto")
    ax.set_yticks(range(K), [f"[MASK] → {a}" for a in cands], fontsize=9)
    ax.set_xticks(range(len(toks)), toks, rotation=90, fontsize=7.5)
    for lab, c in zip(ax.get_xticklabels(), token_colors(toks)):
        lab.set_color(c if c != "#9C9A92" else "#777")
    ax.set_title("③ 推理：最后一层里，每个候选的 [MASK] 在看输入里的哪些词（越深 = 注意力越大）", loc="left", fontsize=11)
    # ④ logit 与概率
    ax = fig.add_subplot(gs[1, 1])
    y = np.arange(K)
    ax.barh(y - 0.2, p, 0.4, color=[C_JEV if i == chosen else "#F0B59B" for i in range(K)], label="模型概率 p")
    ax.barh(y + 0.2, ps, 0.4, color=C_GPT, alpha=0.6, label="专家目标 p*")
    for i in range(K):
        ax.text(max(p[i], ps[i]) + 0.02, i, f"logit {logits[i]:+.1f}  →  p = {p[i]:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y, cands, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("④ [MASK] 向量 → Wx + b → logit → softmax → 概率", loc="left", fontsize=11)
    # ⑤ 输出
    ax = fig.add_subplot(gs[2, :]); ax.axis("off")
    ax.set_title("⑤ 输出", loc="left", fontsize=13, fontweight="bold")
    ok_color = {"ok": "#1D9E75", "success": "#1D9E75", "fail": "#C53030", "disaster": "#C53030"}[result]
    ax.text(0, 0.62, f"按概率采样 → 执行  {action}（p = {p[chosen]:.2f}）", fontsize=14, color=C_JEV,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0, 0.22, f"环境反馈：{RESULT_ZH[result]}", fontsize=13, color=ok_color, transform=ax.transAxes)
    if after is not None and result == "ok":
        same = after.obs_text == s.obs_text
        ax.text(0.5, 0.22, "（观测没有变化：解锁这件事 agent 看不到）" if same else f"新的观测：{after.obs_text}",
                fontsize=12, color=C_OBS, transform=ax.transAxes)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"step": k, "goal": s.goal_text, "obs": s.obs_text, "action": action, "p": float(p[chosen]),
            "result": result, "top": cands[int(np.argmax(p))], "p_top": float(p.max())}


def run_scenario(model, vocab, seed):
    s = State(files=["notes.txt", "report.md", "todo.md"], target="report.md", verb="edit", locked={"report.md"})
    rng = random.Random(seed)
    steps = []
    for _ in range(12):
        b = batch_laya(vocab, [s], DEV)
        with torch.no_grad():
            p = torch.softmax(model(b, 1)[0], -1)[:len(candidates(s))].cpu().tolist()
        a = rng.choices(candidates(s), weights=p)[0]
        before = copy_state(s)
        r = step(s, a)
        steps.append((before, a, r, copy_state(s)))
        if r in ("success", "disaster"):
            break
    return steps


def main():
    model, vocab, _ = load("laya_ce")
    # 找一条“有代表性”的轨迹：先碰到一次打不开，再解锁，最后完成（按概率采样，不是人为指定动作）
    for seed in range(200):
        steps = run_scenario(model, vocab, seed)
        acts = [a for _, a, _, _ in steps]
        results = [r for _, _, r, _ in steps]
        if results[-1] == "success" and "fail" in results and any(a.startswith("unlock") for a in acts) and len(steps) <= 6:
            break
    print(f"场景轨迹（seed {seed}）：", " → ".join(f"{a}[{r}]" for _, a, r, _ in steps))
    d = OUT / "scenario"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (before, a, r, after) in enumerate(steps, 1):
        rows.append(draw_step(k, before, model, vocab, a, r, after, d / f"step_{k}.png"))
        print(f"  → out/scenario/step_{k}.png")
    frames = [Image.open(d / f"step_{k}.png").convert("RGB") for k in range(1, len(steps) + 1)]
    W = max(f.width for f in frames); H = max(f.height for f in frames)
    frames = [f.resize((W, H)) for f in frames]
    frames[0].save(OUT / "scenario.gif", save_all=True, append_images=frames[1:], duration=[3500] * len(frames), loop=0)
    print("  → out/scenario.gif")
    # 总览：每一步的输入 → 模型最看好的动作 → 实际采样的动作 → 结果
    fig, ax = plt.subplots(figsize=(15, 1.2 + 0.9 * len(rows)))
    ax.axis("off")
    cols = ["步", "观测（输入）", "模型最看好的动作", "实际采样执行", "环境反馈"]
    xs = [0.0, 0.04, 0.42, 0.66, 0.86]
    for x, c in zip(xs, cols):
        ax.text(x, 1.0, c, fontsize=12, fontweight="bold", transform=ax.transAxes, va="top")
    for i, r in enumerate(rows):
        y = 0.86 - i * (0.86 / len(rows))
        res_c = "#C53030" if r["result"] in ("fail", "disaster") else "#1D9E75"
        vals = [str(r["step"]), r["obs"], f"{r['top']}（{r['p_top']:.2f}）", f"{r['action']}（{r['p']:.2f}）",
                RESULT_ZH[r["result"]]]
        for x, v, c in zip(xs, vals, ["#333", C_OBS, "#333", C_JEV, res_c]):
            ax.text(x, y, v, fontsize=11, color=c, transform=ax.transAxes, va="top")
    fig.suptitle("实验 7 · 一个完整场景：目标 “edit report.md”，report.md 被锁住（agent 看不到）", fontsize=14, y=1.02)
    fig.savefig(OUT / "7_scenario.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  → out/7_scenario.png")


if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang HK", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    main()
