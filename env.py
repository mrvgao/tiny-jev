"""
TinyFS：一个迷你的“文件系统 agent”环境（照着视频里 open file / list files 的例子做）。

    状态：目录里有哪些文件、agent 已经知道了哪些、当前打开的是哪个；有些文件被锁住，直接打开会失败
    目标：edit / read / delete 某个文件
    候选：由“任务 + 当前状态”映射出来的一组可执行动作（视频里的 Candidates 从哪里来）
    目标概率 p*：一个规则专家在每个状态下对候选动作的概率分布（视频里带 * 的目标值）

关键设计：观测里看不到“文件被锁住”，失败后状态也不变。
所以“总选最高分”的策略遇到锁住的文件会一直重复 open_file，陷入循环；
按概率采样才有机会试到 unlock_file —— 这就是视频里说的探索问题。
"""

import random
from dataclasses import dataclass, field

FILES_TRAIN = ["report.md", "notes.txt", "todo.md", "data.csv", "plan.md", "budget.xlsx",
               "draft.docx", "readme.md", "slides.pptx", "log.txt", "photo.png", "config.yaml"]
FILES_UNSEEN = ["invoice.pdf", "resume.docx", "summary.md", "contacts.csv", "diary.txt", "poster.png"]

VERBS = ["edit", "read", "delete"]
# 测试“换个说法还能不能懂”：训练时只见过左边的动词，测试时换成同义词
PARAPHRASES = {"edit": ["modify", "update", "change"],
               "read": ["view", "check", "review"],
               "delete": ["remove", "erase", "discard"]}


@dataclass
class State:
    files: list           # 目录里真实存在的文件（agent 看不到）
    target: str           # 目标文件
    verb: str             # edit / read / delete
    known: list = field(default_factory=list)   # agent 已经知道的文件（list / search 之后）
    opened: str | None = None
    locked: set = field(default_factory=set)    # 被锁住的文件（agent 看不到）
    unlocked: set = field(default_factory=set)
    goal_word: str | None = None                # 目标里用的动词（默认就是 verb，测试时可换成同义词）

    @property
    def goal_text(self):
        return f"{self.goal_word or self.verb} {self.target}"

    @property
    def obs_text(self):
        known = " ".join(self.known) if self.known else "nothing"
        return f"opened {self.opened or 'none'} ; known {known}"


def stem(name: str) -> str:
    return name.split(".")[0]


def candidates(s: State) -> list:
    """动作空间映射：给定任务和当前观测，列出现在可以执行的动作。"""
    acts = ["list_files", f"search {stem(s.target)}"]
    for f in s.known:
        if f != s.opened:
            acts.append(f"open_file {f}")
        acts += [f"unlock_file {f}", f"delete_file {f}"]
    if s.opened:
        acts += [f"edit_file {s.opened}", f"read_file {s.opened}", "close_file"]
    return acts


def expert(s: State) -> dict:
    """专家的目标概率 p*（只依据 agent 能看到的信息，所以它也不知道文件是否被锁）。"""
    t, w = s.target, {}
    if t not in s.known:                      # 还不知道目标文件在不在：先看看有哪些文件
        w = {"list_files": 0.7, f"search {stem(t)}": 0.3}
    elif s.verb == "delete":
        w = {f"delete_file {t}": 0.9, "list_files": 0.1}
    elif s.opened == t:                       # 已经打开：直接执行对应操作
        w = {f"{s.verb}_file {t}": 0.9, "close_file": 0.1}
    else:                                     # 知道文件但没打开：大概率直接打开，小概率先解锁
        w = {f"open_file {t}": 0.8, f"unlock_file {t}": 0.15, "list_files": 0.05}
        if t in s.unlocked:
            w = {f"open_file {t}": 0.95, "list_files": 0.05}
    cands = candidates(s)
    w = {a: p for a, p in w.items() if a in cands}
    z = sum(w.values())
    return {a: w.get(a, 0.0) / z for a in cands}


def step(s: State, action: str) -> str:
    """执行动作，返回 'ok' / 'fail' / 'success' / 'disaster'。"""
    op, _, arg = action.partition(" ")
    if op == "list_files":
        s.known = list(s.files)
    elif op == "search":
        s.known = sorted(set(s.known) | {f for f in s.files if stem(f) == arg})
    elif op == "open_file":
        if arg in s.locked and arg not in s.unlocked:
            return "fail"                     # 打不开，状态不变
        s.opened = arg
    elif op == "unlock_file":
        s.unlocked.add(arg)
    elif op == "close_file":
        s.opened = None
    elif op in ("edit_file", "read_file"):
        return "success" if (op == f"{s.verb}_file" and arg == s.target) else "fail"
    elif op == "delete_file":
        if s.verb == "delete" and arg == s.target:
            return "success"
        return "disaster"                     # 删错文件：任务失败
    return "ok"


def random_episode_start(rng: random.Random, pool=FILES_TRAIN, p_locked=0.3, goal_words=None) -> State:
    files = rng.sample(pool, rng.randint(3, 6))
    target = rng.choice(files)
    verb = rng.choice(VERBS)
    locked = {target} if rng.random() < p_locked else set()
    goal_word = rng.choice(goal_words[verb]) if goal_words else None
    return State(files=files, target=target, verb=verb, locked=locked, goal_word=goal_word)


def copy_state(s: State) -> State:
    return State(files=list(s.files), target=s.target, verb=s.verb, known=list(s.known), opened=s.opened,
                 locked=set(s.locked), unlocked=set(s.unlocked), goal_word=s.goal_word)


def sample_states(n: int, seed: int = 0, pool=FILES_TRAIN, goal_words=None, max_steps=8) -> list:
    """按专家策略把任务完整走一遍，收集途中经过的每个状态（状态分布和真实使用时一致）。"""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = random_episode_start(rng, pool, goal_words=goal_words)
        seen = set()
        for _ in range(max_steps):
            key = (s.obs_text, s.goal_text)
            if key not in seen:                  # 同一个状态（比如反复打开失败）只收一次
                seen.add(key)
                out.append(copy_state(s))
            p = expert(s)
            a = rng.choices(list(p), weights=list(p.values()))[0]
            if step(s, a) in ("success", "disaster"):
                break
    rng.shuffle(out)
    return out[:n]
