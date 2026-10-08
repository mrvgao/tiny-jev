"""
训练 tiny-Jev 的各个模型。

    python train.py --model laya                      # 从零训练 Laya（交叉熵）
    python train.py --model laya --loss brier         # 换成 Brier 损失
    python train.py --model nanojev
    python train.py --model gpt                       # GPT 生成式基线
    python train.py --model nanojev_qwen              # NanoJev + Qwen3-0.6B（只训练最后 4 层 + 头）
    python train.py --model laya_bert                 # Laya + ModernBERT-base（全量微调）

输出：checkpoints/<name>.pt 和 checkpoints/<name>.json（训练曲线和三个测试集上的指标）
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from data import Vocab, batch_gpt, batch_laya, batch_nanojev, targets
from env import FILES_UNSEEN, PARAPHRASES, expert, sample_states
from models import LOSSES, GPTPolicy, Laya, LayaBert, NanoJev, NanoJevQwen

HERE = Path(__file__).parent
CKPT = HERE / "checkpoints"
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
SCRATCH = {"gpt": GPTPolicy, "nanojev": NanoJev, "laya": Laya}
PRETRAINED = {"nanojev_qwen": NanoJevQwen, "laya_bert": LayaBert}


def datasets(n_train=20000, n_eval=600):
    return {
        "train": sample_states(n_train, seed=0),
        "test": sample_states(n_eval, seed=101),                                   # 同分布
        "paraphrase": sample_states(n_eval, seed=102, goal_words=PARAPHRASES),     # 动词换成没见过的同义词
        "unseen_files": sample_states(n_eval, seed=103, pool=FILES_UNSEEN),        # 文件名都没见过
    }


def jev_logits(kind, model, vocab, states):
    if kind == "nanojev":
        return model(batch_nanojev(vocab, states, DEV), len(states))
    if kind == "laya":
        return model(batch_laya(vocab, states, DEV), len(states))
    return model(states)  # 预训练版本自己处理分词


@torch.no_grad()
def evaluate(kind, model, vocab, states, bs=32):
    """KL(p*‖p)、top-1（预测最高的动作 == 专家最高的动作）、概率平均绝对误差。"""
    model.eval()
    kl = top1 = mae = n_probs = 0.0
    for i in range(0, len(states), bs):
        chunk = states[i:i + bs]
        p_star, valid = targets(chunk, DEV)
        p = torch.softmax(jev_logits(kind, model, vocab, chunk), -1).masked_fill(~valid, 0.0)
        kl += (p_star * (torch.log(p_star.clamp_min(1e-9)) - torch.log(p.clamp_min(1e-9)))).sum().item()
        top1 += (p.argmax(-1) == p_star.argmax(-1)).sum().item()
        mae += ((p - p_star).abs() * valid).sum().item()
        n_probs += valid.sum().item()
    model.train()
    return {"kl": kl / len(states), "top1": top1 / len(states), "mae": mae / n_probs}


@torch.no_grad()
def evaluate_gpt(model, vocab, states):
    """GPT 基线：贪心生成一个动作，看它是不是专家最想做的那个（也统计生成了非法动作的比例）。"""
    from experiments import gpt_generate
    from env import candidates
    ok = invalid = 0
    for s in states:
        a = gpt_generate(model, vocab, s, greedy=True)
        p = expert(s)
        ok += a == max(p, key=p.get)
        invalid += a not in candidates(s)
    return {"top1": ok / len(states), "invalid": invalid / len(states)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SCRATCH) + list(PRETRAINED))
    ap.add_argument("--loss", default="ce", choices=list(LOSSES))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    pretrained = args.model in PRETRAINED
    steps = args.steps or (600 if pretrained else 3000)
    bs = args.batch or (8 if pretrained else 64)
    name = args.name or f"{args.model}_{args.loss}"
    CKPT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    rng = random.Random(0)

    data = datasets()
    vocab = Vocab(data["train"])
    if pretrained:
        model = PRETRAINED[args.model]().to(DEV)
        body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
        opt = torch.optim.AdamW([{"params": body, "lr": 2e-5}, {"params": model.head.parameters(), "lr": 1e-3}])
    else:
        model = SCRATCH[args.model](len(vocab)).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{name}: 可训练 {n_train / 1e6:.2f}M / 总共 {n_all / 1e6:.1f}M 参数，{steps} 步，batch {bs}，设备 {DEV}")

    loss_fn = LOSSES[args.loss]
    history, t0 = [], time.time()
    for step in range(1, steps + 1):
        for g, lr in zip(opt.param_groups, base_lrs):     # warmup + cosine
            g["lr"] = lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        states = rng.sample(data["train"], bs)
        if args.model == "gpt":
            acts = [rng.choices(list(p), weights=list(p.values()))[0] for p in map(expert, states)]  # 按 p* 采样一个动作当答案
            b = batch_gpt(vocab, states, acts, DEV)
            logits = model(b["ids"])
            tgt = b["ids"][:, 1:]
            m = b["loss_mask"][:, 1:]
            loss = torch.nn.functional.cross_entropy(logits[:, :-1][m], tgt[m])
        else:
            p_star, valid = targets(states, DEV)
            loss = loss_fn(jev_logits(args.model, model, vocab, states), p_star, valid)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % (50 if pretrained else 250) == 0 or step == steps:
            rec = {"step": step, "loss": loss.item(), "time": time.time() - t0}
            if args.model != "gpt":
                rec["test"] = evaluate(args.model, model, vocab, data["test"][:200])
            history.append(rec)
            extra = f"  test KL {rec['test']['kl']:.3f}  top1 {rec['test']['top1']:.0%}" if "test" in rec else ""
            print(f"  step {step:5d}  loss {loss.item():.4f}{extra}  ({rec['time']:.0f}s)", flush=True)

    final = {}
    for split in ("test", "paraphrase", "unseen_files"):
        final[split] = evaluate_gpt(model, vocab, data[split][:200]) if args.model == "gpt" else \
            evaluate(args.model, model, vocab, data[split])
        print(f"  [{split}] " + "  ".join(f"{k} {v:.3f}" for k, v in final[split].items()))
    state = {k: v.cpu() for k, v in model.state_dict().items()} if not pretrained else \
        {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
    torch.save({"model": args.model, "loss": args.loss, "vocab": vocab.itos, "state": state}, CKPT / f"{name}.pt")
    (CKPT / f"{name}.json").write_text(json.dumps({"history": history, "final": final, "params_trainable": n_train,
                                                   "params_total": n_all, "train_seconds": time.time() - t0}, indent=1))


if __name__ == "__main__":
    main()
