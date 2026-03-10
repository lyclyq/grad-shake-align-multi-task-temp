# src/hpo.py
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .artifacts import make_run_name, prepare_run_dir, dump_json


TRIALS_HEADER = [
    "trial_id",
    "trial_tag",
    "stage",
    "score",
    "val_max",
    "val_final",
    "val_avg",
    "trial_cfg_json",
    "seeds",
    "run_dir",
    "metrics_csv",
]


# -----------------------------
# Small helpers
# -----------------------------
def _stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_all_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r]


def _append_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = (not csv_path.exists()) or (csv_path.stat().st_size == 0)
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIALS_HEADER)
        if need_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRIALS_HEADER})


def _row_by_trial_tag(rows: List[Dict[str, Any]], trial_tag: str) -> Optional[Dict[str, Any]]:
    for r in rows[::-1]:
        if str(r.get("trial_tag", "")) == str(trial_tag):
            return r
    return None


def _best_row(rows: List[Dict[str, Any]], *, stage_prefix: Optional[str] = None, only_aggregate: bool = False) -> Optional[Dict[str, Any]]:
    best = None
    best_score = float("-inf")
    for r in rows:
        if stage_prefix and not str(r.get("stage", "")).startswith(stage_prefix):
            continue
        if only_aggregate:
            seeds = str(r.get("seeds", "")).strip()
            if "," not in seeds:
                continue
        try:
            s = float(r.get("score", "-inf"))
        except Exception:
            continue
        if s > best_score:
            best, best_score = r, s
    return best


def _flatten_sets(prefix: str, d: Dict[str, Any], out: List[str]) -> None:
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten_sets(p, v, out)
        else:
            out.append(f"{p}={v}")


def _copy_metrics_csv(row: Dict[str, Any], dest: Path) -> bool:
    src = str(row.get("metrics_csv", "")).strip()
    if not src:
        return False
    sp = Path(src)
    if not sp.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sp, dest)
    return True


def _lr_neighbors_one_step(cands: List[float], center: float) -> List[float]:
    if not cands:
        return [float(center)]
    xs = sorted(float(x) for x in cands)
    idx = int(np.argmin(np.abs(np.asarray(xs, dtype=float) - float(center))))
    out: List[float] = [xs[idx]]
    if idx - 1 >= 0:
        out.insert(0, xs[idx - 1])
    if idx + 1 < len(xs):
        out.append(xs[idx + 1])
    dedup: List[float] = []
    seen = set()
    for x in out:
        fx = float(x)
        if fx not in seen:
            dedup.append(fx)
            seen.add(fx)
    return dedup


# -----------------------------
# Run dir + strict invariants
# -----------------------------
def _ensure_hpo_run_dir(cfg: Dict[str, Any]) -> Path:
    """
    Priority:
      1) cfg.io.run_dir + overwrite semantics
      2) cfg.io.root + timestamped make_run_name(kind=hpo)
    """
    io = cfg["io"]
    overwrite = str(io["overwrite"])

    run_dir_cfg = io.get("run_dir", None)
    if run_dir_cfg:
        p = Path(str(run_dir_cfg))
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.exists():
            if overwrite == "resume":
                return p
            if overwrite == "force":
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                return p
            ans = input(f"[artifacts] {p} exists. Overwrite? [y/N] ").strip().lower()
            if ans in {"y", "yes"}:
                shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
                return p
            return p

        p.mkdir(parents=True, exist_ok=True)
        return p

    root = io["root"]
    run_name = make_run_name(cfg, extra={"kind": "hpo"})
    run_dir, _ = prepare_run_dir(root=root, run_name=run_name, overwrite=overwrite)
    return Path(run_dir)


def _assert_capacity_unified(cfg: Dict[str, Any]) -> None:
    """
    Hard command:
      baseline_r.rank == ours.r
      baseline_R.rank == ours.R
    """
    br = int(cfg["method"]["baseline_r"]["lora"]["r"])
    bR = int(cfg["method"]["baseline_R"]["lora"]["r"])
    orr = int(cfg["method"]["ours"]["lora"]["r"])
    oRR = int(cfg["method"]["ours"]["lora"]["R"])
    if br != orr:
        raise RuntimeError(f"[HPO][CapacityMismatch] baseline_r.r={br} != ours.r={orr}")
    if bR != oRR:
        raise RuntimeError(f"[HPO][CapacityMismatch] baseline_R.r={bR} != ours.R={oRR}")


def _baseline_tags(cfg: Dict[str, Any]) -> List[str]:
    hpo = cfg["hpo"]
    variants = hpo["baseline_variants"]
    if not isinstance(variants, list) or not variants:
        raise RuntimeError("[HPO] missing hpo.baseline_variants (must be list of {tag: ...})")
    tags: List[str] = []
    for i, v in enumerate(variants):
        if not isinstance(v, dict) or "tag" not in v:
            raise RuntimeError(f"[HPO] baseline_variants[{i}] must be dict with tag")
        tag = str(v["tag"])
        if tag not in {"baseline_r", "baseline_R"}:
            raise RuntimeError(f"[HPO] baseline_variants[{i}].tag must be baseline_r/baseline_R, got {tag}")
        tags.append(tag)
    out: List[str] = []
    seen = set()
    for t in tags:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _baseline_lora_from_method(cfg: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """
    STRICT: baseline truth only from method.baseline_{r,R}.lora
    """
    if tag not in {"baseline_r", "baseline_R"}:
        raise ValueError(tag)
    blk = cfg["method"][tag]["lora"]
    r = int(blk["r"])
    alpha = float(blk["alpha"])
    dropout = float(blk["dropout"])
    return {"r": r, "alpha": alpha, "dropout": dropout}


# -----------------------------
# plan_status.json (explicit)
# -----------------------------
@dataclass
class PlanItem:
    idx: int
    stage: str
    trial_tag: str
    seed: Optional[int]  # None for aggregate rows
    run_dir: str
    override: Dict[str, Any]
    status: str = "pending"  # pending | running | done | failed
    tries: int = 0
    last_rc: Optional[int] = None
    last_update: int = 0


def _load_plan_status(path: Path) -> Dict[str, Any]:
    obj = _read_json(path, default=None)
    if not isinstance(obj, dict) or "items" not in obj:
        return {"items": [], "created_at": int(time.time()), "updated_at": int(time.time())}
    if not isinstance(obj.get("items", None), list):
        obj["items"] = []
    return obj


def _save_plan_status(path: Path, st: Dict[str, Any]) -> None:
    st2 = dict(st)
    st2["updated_at"] = int(time.time())
    _atomic_write_json(path, st2)


def _plan_index(st: Dict[str, Any]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    items = st.get("items", [])
    if not isinstance(items, list):
        return idx
    for i, it in enumerate(items):
        tt = str(it.get("trial_tag", ""))
        sd = it.get("seed", None)
        key = f"{tt}__seed_{sd}" if sd is not None else f"{tt}__agg"
        idx[key] = i
    return idx


def _update_item(st: Dict[str, Any], item: PlanItem) -> None:
    items = st.get("items", [])
    if not isinstance(items, list):
        items = []
        st["items"] = items

    key = f"{item.trial_tag}__seed_{item.seed}" if item.seed is not None else f"{item.trial_tag}__agg"
    index = _plan_index(st)
    payload = {
        "idx": int(item.idx),
        "stage": str(item.stage),
        "trial_tag": str(item.trial_tag),
        "seed": item.seed,
        "run_dir": str(item.run_dir),
        "override": item.override,
        "status": str(item.status),
        "tries": int(item.tries),
        "last_rc": item.last_rc,
        "last_update": int(item.last_update),
    }
    if key in index:
        items[index[key]] = payload
    else:
        items.append(payload)


def _is_done_in_trials_csv(trials_csv: Path, trial_tag: str) -> bool:
    rows = _read_all_rows(trials_csv)
    return _row_by_trial_tag(rows, trial_tag) is not None


def _next_pending_item(st: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    items = st.get("items", [])
    if not isinstance(items, list):
        return None
    cand = [it for it in items if str(it.get("status", "")) in {"pending", "failed"}]
    if not cand:
        return None
    cand.sort(key=lambda it: int(it.get("idx", 10**9)))
    return cand[0]


def _next_pending_item_by_stage(st: Dict[str, Any], stages: List[str]) -> Optional[Dict[str, Any]]:
    items = st.get("items", [])
    if not isinstance(items, list):
        return None
    stage_set = {str(s) for s in stages}
    cand = [
        it
        for it in items
        if str(it.get("status", "")) in {"pending", "failed"} and str(it.get("stage", "")) in stage_set
    ]
    if not cand:
        return None
    cand.sort(key=lambda it: int(it.get("idx", 10**9)))
    return cand[0]


# -----------------------------
# runner invocation
# -----------------------------
def _run_one_trial(
    *,
    base_config_path: str,
    schedule_path: Optional[str],
    base_set_args: List[str],
    trial_override: Dict[str, Any],
    trial_tag: str,
    stage: str,
    hpo_run_dir: Path,
    score_weights: Tuple[float, float, float],
    run_dir: Optional[Path] = None,
    overwrite_mode: str = "resume",
) -> int:
    """
    Returns subprocess returncode.
    NOTE: trials.csv is appended by runner ONLY if training finishes.
    """
    cmd = [sys.executable, "scripts/run.py", "train", "--config", base_config_path]
    if schedule_path:
        cmd += ["--schedule", schedule_path]

    sets: List[str] = []
    if base_set_args:
        sets.extend(base_set_args)

    if run_dir is not None:
        sets.append(f"io.run_dir={str(run_dir)}")
        sets.append(f"io.overwrite={overwrite_mode}")

    trial_sets: List[str] = []
    _flatten_sets("", trial_override, trial_sets)
    sets.extend(trial_sets)

    for s in sets:
        cmd += ["--set", s]
    cmd += ["--trial_tag", trial_tag]

    env = os.environ.copy()
    env["GSA_HPO_RUN_DIR"] = str(hpo_run_dir)
    env["GSA_STAGE"] = stage
    w_max, w_final, w_avg = score_weights
    env["GSA_W_MAX"] = str(w_max)
    env["GSA_W_FINAL"] = str(w_final)
    env["GSA_W_AVG"] = str(w_avg)

    print(" ".join(cmd))
    p = subprocess.run(cmd, check=False, env=env)
    return int(p.returncode)


# -----------------------------
# seed aggregation
# -----------------------------
def _aggregate_seed_trial(
    *,
    trials_csv: Path,
    base_trial_tag: str,
    stage: str,
    seeds: List[int],
    trial_cfg_json: str,
) -> Optional[Dict[str, Any]]:
    rows = _read_all_rows(trials_csv)
    seed_rows: List[Dict[str, Any]] = []
    for sd in seeds:
        tag = f"{base_trial_tag}__s{sd}"
        r = _row_by_trial_tag(rows, tag)
        if r is not None:
            seed_rows.append(r)
    if len(seed_rows) != len(seeds):
        return None

    def _f(x, default=-1e18):
        try:
            return float(x)
        except Exception:
            return float(default)

    score_mean = float(np.mean([_f(r.get("score")) for r in seed_rows]))
    val_max_mean = float(np.mean([_f(r.get("val_max")) for r in seed_rows]))
    val_final_mean = float(np.mean([_f(r.get("val_final")) for r in seed_rows]))
    val_avg_mean = float(np.mean([_f(r.get("val_avg")) for r in seed_rows]))

    best_seed_row = max(seed_rows, key=lambda r: _f(r.get("score")))
    agg_row = {
        "trial_id": "",
        "trial_tag": base_trial_tag,
        "stage": stage,
        "score": score_mean,
        "val_max": val_max_mean,
        "val_final": val_final_mean,
        "val_avg": val_avg_mean,
        "trial_cfg_json": trial_cfg_json,
        "seeds": ",".join([str(x) for x in seeds]),
        "run_dir": str(best_seed_row.get("run_dir", "")),
        "metrics_csv": str(best_seed_row.get("metrics_csv", "")),
    }

    if _row_by_trial_tag(rows, base_trial_tag) is None:
        _append_row(trials_csv, agg_row)

    return agg_row


# -----------------------------
# Grid solve (drop + density)
# -----------------------------
@dataclass
class KnobRange:
    name: str
    kind: str  # float | choice
    lo: Optional[float] = None
    hi: Optional[float] = None
    choices: Optional[List[float]] = None
    weight: float = 0.0
    m: int = 0


def _solve_grid_points(*, L: int, B: int, knobs: List[KnobRange], max_m_float: int = 7, max_m_choice: int = 0) -> Tuple[List[KnobRange], Dict[str, Any]]:
    knobs = [k for k in knobs]
    knobs = sorted(knobs, key=lambda k: float(k.weight), reverse=True)

    def m_min(k: KnobRange) -> int:
        if k.kind == "choice":
            if not k.choices:
                return 0
            return min(2, len(k.choices))
        return 2

    def m_max(k: KnobRange) -> int:
        if k.kind == "choice":
            if not k.choices:
                return 0
            if max_m_choice and max_m_choice > 0:
                return min(int(max_m_choice), len(k.choices))
            return len(k.choices)
        return int(max(2, max_m_float))

    def total_points(ks: List[KnobRange]) -> int:
        prod = 1
        for kk in ks:
            prod *= max(1, int(kk.m))
        return int(L * prod)

    for k in knobs:
        k.m = m_min(k)

    dropped: List[str] = []
    while knobs and total_points(knobs) > B:
        tail = knobs.pop(-1)
        dropped.append(tail.name)

    if not knobs:
        meta = {"dropped_knobs": dropped, "reason": "budget_too_small_even_after_drop"}
        return [], meta

    def score_gain(k: KnobRange) -> float:
        mm = max(1, int(k.m))
        return float(k.weight) / float((mm + 1) / mm)

    changed = True
    while changed:
        changed = False
        cand = sorted(knobs, key=lambda k: score_gain(k), reverse=True)
        for k in cand:
            if k.m >= m_max(k):
                continue
            old_m = int(k.m)
            k.m = old_m + 1
            if total_points(knobs) <= B:
                changed = True
                break
            k.m = old_m

    meta = {
        "dropped_knobs": dropped,
        "L": int(L),
        "B": int(B),
        "final_total_points": int(total_points(knobs)),
        "per_knob_m": {k.name: int(k.m) for k in knobs},
    }
    return knobs, meta


def _make_knob_values(k: KnobRange) -> List[float]:
    if k.kind == "choice":
        ch = list(k.choices or [])
        if not ch:
            return []
        if k.m <= 0 or k.m >= len(ch):
            return [float(x) for x in ch]
        idxs = np.linspace(0, len(ch) - 1, num=int(k.m)).round().astype(int).tolist()
        out: List[float] = []
        for ii in idxs:
            v = float(ch[int(ii)])
            if v not in out:
                out.append(v)
        return out

    if k.lo is None or k.hi is None:
        raise RuntimeError(f"[HPO] float knob range missing bounds for {k.name}")
    lo = float(k.lo)
    hi = float(k.hi)
    m = int(max(2, k.m))
    vals = np.linspace(lo, hi, num=m).astype(float).tolist()
    out: List[float] = []
    for v in vals:
        v = float(v)
        if v not in out:
            out.append(v)
    return out


# -----------------------------
# Bayes optimizer (GP + EI)
# -----------------------------
def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length: float, var: float) -> np.ndarray:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    s1 = np.sum(X1 * X1, axis=1, keepdims=True)
    s2 = np.sum(X2 * X2, axis=1, keepdims=True).T
    D2 = s1 + s2 - 2.0 * (X1 @ X2.T)
    return var * np.exp(-0.5 * D2 / max(1e-12, length * length))


def _gp_posterior(
    X: np.ndarray,
    y: np.ndarray,
    Xcand: np.ndarray,
    *,
    length: float = 1.0,
    var: float = 1.0,
    noise: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    Xcand = np.asarray(Xcand, dtype=float)

    n = X.shape[0]
    if n == 0:
        mu0 = np.zeros((Xcand.shape[0],), dtype=float)
        sg0 = np.ones((Xcand.shape[0],), dtype=float)
        return mu0, sg0

    K = _rbf_kernel(X, X, length=length, var=var)
    K = K + float(noise) * np.eye(n, dtype=float)
    Ks = _rbf_kernel(X, Xcand, length=length, var=var)
    Kss = _rbf_kernel(Xcand, Xcand, length=length, var=var)

    L = np.linalg.cholesky(K + 1e-12 * np.eye(n))
    v = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, v)

    mu = (Ks.T @ alpha).reshape(-1)

    w = np.linalg.solve(L, Ks)
    cov = Kss - (w.T @ w)
    cov = (cov + cov.T) * 0.5
    var_diag = np.clip(np.diag(cov), 1e-12, None)
    sigma = np.sqrt(var_diag)
    return mu, sigma


def _phi(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x)


def _Phi(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    imp = mu - best - float(xi)
    Z = imp / np.clip(sigma, 1e-12, None)
    ei = imp * _Phi(Z) + sigma * _phi(Z)
    ei[sigma < 1e-12] = 0.0
    return ei


def _encode_config(
    *,
    lr: float,
    knobs: Dict[str, float],
    knob_space: List[KnobRange],
    choice_maps: Dict[str, List[float]],
    lr_min: float,
    lr_max: float,
) -> List[float]:
    v: List[float] = []
    lrl = math.log10(max(1e-20, float(lr)))
    mn = math.log10(max(1e-20, float(lr_min)))
    mx = math.log10(max(1e-20, float(lr_max)))
    if abs(mx - mn) < 1e-12:
        v.append(0.5)
    else:
        v.append((lrl - mn) / (mx - mn))

    for k in knob_space:
        name = k.name
        if k.kind == "choice":
            ch = choice_maps.get(name, list(k.choices or []))
            if not ch:
                v.append(0.0)
            else:
                try:
                    idx = int(np.argmin(np.abs(np.array(ch, dtype=float) - float(knobs.get(name, ch[0])))))
                except Exception:
                    idx = 0
                if len(ch) == 1:
                    v.append(0.0)
                else:
                    v.append(float(idx) / float(len(ch) - 1))
        else:
            lo = float(k.lo if k.lo is not None else 0.0)
            hi = float(k.hi if k.hi is not None else lo + 1.0)
            x = float(knobs.get(name, lo))
            if abs(hi - lo) < 1e-12:
                v.append(0.5)
            else:
                v.append((x - lo) / (hi - lo))
    return [float(max(0.0, min(1.0, x))) for x in v]


def _random_sample_config(
    rng: np.random.Generator,
    *,
    lr_candidates: List[float],
    knob_space: List[KnobRange],
) -> Tuple[float, Dict[str, float]]:
    lr = float(rng.choice(lr_candidates))
    assign: Dict[str, float] = {}
    for k in knob_space:
        if k.kind == "choice":
            ch = list(k.choices or [])
            if not ch:
                continue
            assign[k.name] = float(rng.choice(ch))
        else:
            if k.lo is None or k.hi is None:
                raise RuntimeError(f"[HPO] float knob range missing bounds for {k.name}")
            lo = float(k.lo)
            hi = float(k.hi)
            assign[k.name] = float(rng.uniform(lo, hi))
    return lr, assign


# -----------------------------
# Alpha probe helpers
# -----------------------------
def _median_from_knob_range(k: KnobRange) -> float:
    if k.kind == "choice":
        ch = list(k.choices or [])
        if not ch:
            return 0.0
        ch2 = sorted([float(x) for x in ch])
        return float(ch2[len(ch2) // 2])
    lo = float(k.lo if k.lo is not None else 0.0)
    hi = float(k.hi if k.hi is not None else lo)
    return float((lo + hi) / 2.0)


def _median_from_spec(spec: Dict[str, Any]) -> float:
    if "kind" not in spec:
        raise RuntimeError("[HPO] knob spec missing key: kind")
    kind = str(spec["kind"])
    if kind == "choice":
        choices = spec.get("choices", None)
        if isinstance(choices, list) and choices:
            ch = sorted([float(x) for x in choices])
            return float(ch[len(ch) // 2])
        raise RuntimeError("[HPO] choice knob spec must provide non-empty choices")
    lo = float(spec["lo"])
    hi = float(spec["hi"])
    return float((lo + hi) / 2.0)


def _score_mean_for_tags(trials_csv: Path, tags: List[str]) -> Optional[float]:
    rows = _read_all_rows(trials_csv)
    vals: List[float] = []
    for t in tags:
        r = _row_by_trial_tag(rows, t)
        if r is None:
            return None
        try:
            vals.append(float(r.get("score", "-inf")))
        except Exception:
            return None
    if not vals:
        return None
    return float(np.mean(np.array(vals, dtype=float)))


# -----------------------------
# Coordinate-descent HPO
# -----------------------------
@dataclass
class ParamSpec:
    key: str
    path: str
    scale: str
    lower: float
    center: float
    upper: float


def _cfg_get_path(obj: Dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in str(dotted).split("."):
        if not isinstance(cur, dict):
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def _cfg_set_path(obj: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = obj
    parts = str(dotted).split(".")
    for part in parts[:-1]:
        nxt = cur.get(part, None)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _safe_stage_slug(x: str) -> str:
    out = []
    for ch in str(x):
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    s = "".join(out).strip("_")
    return s or "param"


def _half_up_div(num: int, den: int) -> int:
    num = int(num)
    den = max(1, int(den))
    return max(1, int(math.floor(float(num) / float(den) + 0.5)))


def _trans(x: float, scale: str) -> float:
    if scale == "log":
        return float(math.log10(max(1e-20, float(x))))
    return float(x)


def _inv_trans(x: float, scale: str) -> float:
    if scale == "log":
        return float(10.0 ** float(x))
    return float(x)


def _make_anchor_grid(*, count: int, lower: float, center: float, upper: float, scale: str, rng: np.random.Generator) -> List[float]:
    lower = float(lower)
    center = float(center)
    upper = float(upper)
    if lower > upper:
        lower, upper = upper, lower
    center = min(max(center, lower), upper)
    count = max(1, int(count))

    if count == 1:
        return [float(center)]
    if count == 2:
        boundary = float(rng.choice([lower, upper]))
        out = [float(center), boundary]
        if out[0] == out[1]:
            out = [float(center), float(lower if boundary != lower else upper)]
        return out

    t_lo = _trans(lower, scale)
    t_ce = _trans(center, scale)
    t_hi = _trans(upper, scale)
    extra = max(0, count - 3)

    span_left = max(1e-12, abs(t_ce - t_lo))
    span_right = max(1e-12, abs(t_hi - t_ce))
    frac_left = span_left / (span_left + span_right)
    extra_left = int(round(extra * frac_left))
    extra_left = max(0, min(extra, extra_left))
    extra_right = extra - extra_left

    left_raw = np.linspace(t_lo, t_ce, num=extra_left + 2).tolist()[:-1]
    right_raw = np.linspace(t_ce, t_hi, num=extra_right + 2).tolist()[1:]
    vals_t = left_raw + [t_ce] + right_raw

    out: List[float] = []
    for v in vals_t:
        x = float(_inv_trans(v, scale))
        x = min(max(x, lower), upper)
        if not out or abs(out[-1] - x) > 1e-15:
            out.append(x)

    anchors = [float(lower), float(center), float(upper)]
    for a in anchors:
        if all(abs(a - z) > 1e-15 for z in out):
            out.append(float(a))
    out = sorted({float(x) for x in out})

    while len(out) > count:
        removable = [i for i, x in enumerate(out) if all(abs(x - a) > 1e-15 for a in anchors)]
        if not removable:
            break
        best_i = removable[len(removable) // 2]
        out.pop(best_i)

    if len(out) < count:
        dense = np.linspace(t_lo, t_hi, num=max(count * 8, count + 2)).tolist()
        for v in dense:
            x = float(_inv_trans(v, scale))
            x = min(max(x, lower), upper)
            if all(abs(x - z) > 1e-15 for z in out):
                out.append(x)
            if len(out) >= count:
                break
        out = sorted(out)

    if len(out) > count:
        out = out[:count]
    return [float(x) for x in out]


def _param_local_grid(*, best_value: float, previous_values: List[float], lower: float, upper: float, scale: str, radix: int) -> List[float]:
    radix = max(1, int(radix))
    if radix == 1:
        return [float(best_value)]

    prev_sorted = sorted({float(x) for x in previous_values})
    t_best = _trans(best_value, scale)
    t_prev = [_trans(x, scale) for x in prev_sorted]
    nearest = [abs(tp - t_best) for tp in t_prev if abs(tp - t_best) > 1e-15]
    if nearest:
        base_step = min(nearest) * 0.5
    else:
        base_step = max(abs(_trans(upper, scale) - _trans(lower, scale)) / 8.0, 1e-6)

    radius = radix // 2
    offsets = list(range(-radius, radius + 1))
    if len(offsets) > radix:
        offsets = offsets[:radix]
    if len(offsets) < radix:
        while len(offsets) < radix:
            offsets.append(offsets[-1] + 1)

    vals: List[float] = []
    for off in offsets:
        tv = t_best + float(off) * base_step
        x = _inv_trans(tv, scale)
        x = min(max(float(x), float(lower)), float(upper))
        vals.append(float(x))

    out = sorted({float(x) for x in vals})
    if all(abs(float(best_value) - x) > 1e-15 for x in out):
        out.append(float(best_value))
        out = sorted(out)
    return out


def _spec_from_cfg(cfg: Dict[str, Any], key: str, tag: Optional[str] = None) -> ParamSpec:
    spec_root = _cfg_get_path(cfg, "hpo.coord.param_specs")
    raw = json.loads(json.dumps(spec_root[key]))
    if "path" in raw:
        path = str(raw["path"])
    else:
        path = str(raw["path_template"]).format(tag=str(tag or ""))
    return ParamSpec(
        key=str(key),
        path=path,
        scale=str(raw["scale"]).strip().lower(),
        lower=float(raw["lower"]),
        center=float(raw["center"]),
        upper=float(raw["upper"]),
    )


def _method_param_specs(cfg: Dict[str, Any], *, family: str, tag: Optional[str] = None) -> List[ParamSpec]:
    coord = _cfg_get_path(cfg, "hpo.coord")
    shared_order = [str(x) for x in coord["shared_order"]]
    method_orders = coord["method_orders"]
    private_order = [str(x) for x in method_orders[family]]
    ordered = shared_order + private_order
    return [_spec_from_cfg(cfg, x, tag=tag) for x in ordered]


def _method_base_override(cfg: Dict[str, Any], *, family: str, tag: Optional[str] = None, stage_epochs: int, stage_max_steps: int) -> Dict[str, Any]:
    tr = {
        "epochs": int(stage_epochs),
        "max_steps": int(stage_max_steps),
    }
    if family == "ours":
        return {
            "method": {"name": "ours", "ours": json.loads(json.dumps(cfg["method"]["ours"]))},
            "train": tr,
        }

    assert tag is not None
    blk = json.loads(json.dumps(cfg["method"][tag]))
    blk["grad_solver"] = "cagrad" if family == "cagrad" else "avg"
    return {
        "method": {"name": tag, tag: blk},
        "train": tr,
    }


def _apply_centers(override: Dict[str, Any], specs: List[ParamSpec]) -> None:
    for sp in specs:
        _cfg_set_path(override, sp.path, float(sp.center))


def _run_seed_aggregate(
    *,
    trials_csv: Path,
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
    hpo_run_dir: Path,
    score_weights: Tuple[float, float, float],
    seed_stage: str,
    agg_stage: str,
    base_tag: str,
    override_seedless: Dict[str, Any],
    seeds: List[int],
    run_root: Path,
) -> Dict[str, Any]:
    for sd in seeds:
        trial_tag = f"{base_tag}__s{sd}"
        if not _is_done_in_trials_csv(trials_csv, trial_tag):
            ov = json.loads(json.dumps(override_seedless))
            ov.setdefault("train", {})
            ov["train"]["seed"] = int(sd)
            rc = _run_one_trial(
                base_config_path=base_config_path,
                schedule_path=schedule_path,
                base_set_args=set_args,
                trial_override=ov,
                trial_tag=trial_tag,
                stage=seed_stage,
                hpo_run_dir=hpo_run_dir,
                score_weights=score_weights,
                run_dir=run_root / f"s{sd}",
                overwrite_mode="resume",
            )
            if rc != 0 and (not _is_done_in_trials_csv(trials_csv, trial_tag)):
                raise RuntimeError(f"[HPO] trial failed: stage={seed_stage} tag={trial_tag} rc={rc}")

    if not _is_done_in_trials_csv(trials_csv, base_tag):
        row = _aggregate_seed_trial(
            trials_csv=trials_csv,
            base_trial_tag=base_tag,
            stage=agg_stage,
            seeds=seeds,
            trial_cfg_json=json.dumps(override_seedless, sort_keys=True),
        )
        if row is None:
            raise RuntimeError(f"[HPO] failed to aggregate seeds for {base_tag}")

    rows = _read_all_rows(trials_csv)
    agg = _row_by_trial_tag(rows, base_tag)
    if agg is None:
        raise RuntimeError(f"[HPO] missing aggregate row for {base_tag}")
    return agg


def _select_best(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = None
    best_score = float("-inf")
    for c in cands:
        try:
            s = float(c["row"].get("score", "-inf"))
        except Exception:
            s = float("-inf")
        if best is None or s > best_score:
            best = c
            best_score = s
    if best is None:
        raise RuntimeError("[HPO] empty candidate list")
    return best


def _score_variance(rows: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for r in rows:
        try:
            vals.append(float(r.get("score", "-inf")))
        except Exception:
            continue
    if len(vals) <= 1:
        return 0.0
    return float(np.var(np.asarray(vals, dtype=float)))


def _run_coordinate_descent(
    *,
    cfg: Dict[str, Any],
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
    hpo_run_dir: Path,
    trials_csv: Path,
    score_weights: Tuple[float, float, float],
    rng: np.random.Generator,
    family: str,
    tag: Optional[str],
    budget_trials: int,
    seeds: List[int],
    stage_epochs: int,
    stage_max_steps: int,
) -> Dict[str, Any]:
    specs = _method_param_specs(cfg, family=family, tag=tag)
    if not specs:
        raise RuntimeError(f"[HPO] no params for family={family} tag={tag}")

    current = _method_base_override(cfg, family=family, tag=tag, stage_epochs=stage_epochs, stage_max_steps=stage_max_steps)
    _apply_centers(current, specs)

    points_per_param = _half_up_div(int(budget_trials), len(specs))
    points_per_param = max(1, int(points_per_param))

    sweeps: List[Dict[str, Any]] = []
    current_best_row: Optional[Dict[str, Any]] = None

    for pi, sp in enumerate(specs):
        values = _make_anchor_grid(
            count=points_per_param,
            lower=sp.lower,
            center=float(_cfg_get_path(current, sp.path)),
            upper=sp.upper,
            scale=sp.scale,
            rng=rng,
        )
        candidates: List[Dict[str, Any]] = []
        for vi, val in enumerate(values):
            ov = json.loads(json.dumps(current))
            _cfg_set_path(ov, sp.path, float(val))
            base_tag = f"{family}__{tag or 'ours'}__cd__p{pi:02d}__{_safe_stage_slug(sp.key)}__v{vi:02d}"
            stage_seed = "baseline_search" if family == "baseline" else "baseline_cagrad_search" if family == "cagrad" else "coord_search"
            stage_agg = "baseline_search.agg" if family == "baseline" else "baseline_cagrad_search.agg" if family == "cagrad" else "coord_search.agg"
            row = _run_seed_aggregate(
                trials_csv=trials_csv,
                base_config_path=base_config_path,
                schedule_path=schedule_path,
                set_args=set_args,
                hpo_run_dir=hpo_run_dir,
                score_weights=score_weights,
                seed_stage=stage_seed,
                agg_stage=stage_agg,
                base_tag=base_tag,
                override_seedless=ov,
                seeds=seeds,
                run_root=hpo_run_dir / "trial_runs" / family / (tag or "ours") / "coord" / f"p{pi:02d}_{_safe_stage_slug(sp.key)}" / f"v{vi:02d}",
            )
            candidates.append({"value": float(val), "override": ov, "row": row})

        best = _select_best(candidates)
        current = json.loads(json.dumps(best["override"]))
        current_best_row = dict(best["row"])
        sweeps.append(
            {
                "param_key": sp.key,
                "param_path": sp.path,
                "scale": sp.scale,
                "lower": float(sp.lower),
                "upper": float(sp.upper),
                "chosen_value": float(best["value"]),
                "candidate_values": [float(c["value"]) for c in candidates],
                "score_variance": float(_score_variance([c["row"] for c in candidates])),
                "best_score": float(best["row"].get("score", "-inf")),
                "best_trial_tag": str(best["row"].get("trial_tag", "")),
            }
        )

    if current_best_row is None:
        raise RuntimeError(f"[HPO] coordinate descent produced no result for family={family} tag={tag}")

    return {
        "family": family,
        "tag": tag,
        "final_override": current,
        "best_row": current_best_row,
        "points_per_param": int(points_per_param),
        "sweeps": sweeps,
    }


def _run_local_refine(
    *,
    cfg: Dict[str, Any],
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
    hpo_run_dir: Path,
    trials_csv: Path,
    score_weights: Tuple[float, float, float],
    family: str,
    tag: Optional[str],
    coord_result: Dict[str, Any],
    seeds: List[int],
    stage_epochs: int,
    stage_max_steps: int,
    top_k: int,
    radix: int,
) -> Dict[str, Any]:
    sweeps = list(coord_result["sweeps"])
    ranked = sorted(sweeps, key=lambda x: float(x.get("score_variance", 0.0)), reverse=True)
    top = ranked[: max(0, int(top_k))]
    if not top:
        return {"enabled": False, "best_row": coord_result["best_row"], "best_override": coord_result["final_override"], "top_params": []}

    grids: List[Tuple[Dict[str, Any], List[float]]] = []
    for sw in top:
        grids.append(
            (
                sw,
                _param_local_grid(
                    best_value=float(sw["chosen_value"]),
                    previous_values=[float(x) for x in sw["candidate_values"]],
                    lower=float(sw["lower"]),
                    upper=float(sw["upper"]),
                    scale=str(sw["scale"]),
                    radix=int(radix),
                ),
            )
        )

    base_override = json.loads(json.dumps(coord_result["final_override"]))
    best_row = coord_result["best_row"]
    best_override = base_override
    emitted = 0
    keys = [g[0]["param_path"] for g in grids]
    values_product = itertools.product(*[g[1] for g in grids])
    for combo_idx, vals in enumerate(values_product):
        ov = json.loads(json.dumps(base_override))
        assign = {}
        for path, val in zip(keys, vals):
            _cfg_set_path(ov, path, float(val))
            assign[path] = float(val)
        base_tag = f"{family}__{tag or 'ours'}__local__{combo_idx:03d}__{_stable_hash(assign)}"
        row = _run_seed_aggregate(
            trials_csv=trials_csv,
            base_config_path=base_config_path,
            schedule_path=schedule_path,
            set_args=set_args,
            hpo_run_dir=hpo_run_dir,
            score_weights=score_weights,
            seed_stage="local_refine",
            agg_stage="local_refine.agg",
            base_tag=base_tag,
            override_seedless=ov,
            seeds=seeds,
            run_root=hpo_run_dir / "trial_runs" / family / (tag or "ours") / "local_refine" / f"combo{combo_idx:03d}",
        )
        emitted += 1
        try:
            if float(row.get("score", "-inf")) > float(best_row.get("score", "-inf")):
                best_row = row
                best_override = ov
        except Exception:
            pass

    return {
        "enabled": True,
        "best_row": best_row,
        "best_override": best_override,
        "top_params": [
            {
                "param_key": str(sw["param_key"]),
                "param_path": str(sw["param_path"]),
                "score_variance": float(sw["score_variance"]),
                "local_values": [float(x) for x in vals],
            }
            for sw, vals in grids
        ],
        "configs": int(emitted),
    }


def _run_bayes_after_refine(
    *,
    cfg: Dict[str, Any],
    base_config_path: str,
    schedule_path: Optional[str],
    set_args: List[str],
    hpo_run_dir: Path,
    trials_csv: Path,
    score_weights: Tuple[float, float, float],
    family: str,
    tag: Optional[str],
    coord_result: Dict[str, Any],
    refine_result: Dict[str, Any],
    seeds: List[int],
    stage_epochs: int,
    stage_max_steps: int,
    trials_budget: int,
) -> Dict[str, Any]:
    if not bool(cfg["hpo"]["use_bayes"]) or int(trials_budget) <= 0:
        return {"enabled": False}

    top_params = list(refine_result.get("top_params", []) or [])
    if not top_params:
        return {"enabled": False}

    knob_space: List[KnobRange] = []
    path_to_scale: Dict[str, str] = {}
    for tp in top_params:
        path = str(tp["param_path"])
        sw = next(x for x in coord_result["sweeps"] if str(x["param_path"]) == path)
        knob_space.append(
            KnobRange(
                name=path,
                kind="float",
                lo=float(sw["lower"]),
                hi=float(sw["upper"]),
                weight=float(sw["score_variance"]),
            )
        )
        path_to_scale[path] = str(sw["scale"])

    def _encode(assign: Dict[str, float]) -> List[float]:
        out: List[float] = []
        for k in knob_space:
            lo = _trans(float(k.lo), path_to_scale[k.name])
            hi = _trans(float(k.hi), path_to_scale[k.name])
            x = _trans(float(assign[k.name]), path_to_scale[k.name])
            if abs(hi - lo) < 1e-12:
                out.append(0.5)
            else:
                out.append((x - lo) / (hi - lo))
        return out

    rows = _read_all_rows(trials_csv)
    X_data: List[List[float]] = []
    y_data: List[float] = []
    seen = set()
    for r in rows:
        if str(r.get("stage", "")) != "local_refine.agg":
            continue
        tt = str(r.get("trial_tag", ""))
        if not tt.startswith(f"{family}__{tag or 'ours'}__local__"):
            continue
        cfgj = json.loads(str(r.get("trial_cfg_json", "{}")))
        assign: Dict[str, float] = {}
        ok = True
        for tp in top_params:
            path = str(tp["param_path"])
            try:
                assign[path] = float(_cfg_get_path(cfgj, path))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        key = json.dumps(assign, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        X_data.append(_encode(assign))
        y_data.append(float(r.get("score", "-inf")))

    if not X_data:
        return {"enabled": False}

    X = np.asarray(X_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    grid_cfg = cfg["hpo"]["grid"]
    bo_rng = np.random.default_rng(int(grid_cfg["bayes_rng_seed"]))
    pool = int(grid_cfg["bayes_pool"])
    length = float(grid_cfg["bayes_gp_length"])
    var = float(grid_cfg["bayes_gp_var"])
    noise = float(grid_cfg["bayes_gp_noise"])
    xi = float(grid_cfg["bayes_ei_xi"])

    base_override = json.loads(json.dumps(refine_result["best_override"]))
    best_row = refine_result["best_row"]
    best_override = base_override
    proposals: List[Dict[str, Any]] = []

    for t in range(int(trials_budget)):
        pool_assigns: List[Dict[str, float]] = []
        pool_vecs: List[List[float]] = []
        for _ in range(pool):
            assign: Dict[str, float] = {}
            for k in knob_space:
                lo = float(k.lo)
                hi = float(k.hi)
                scale = path_to_scale[k.name]
                if scale == "log":
                    tv = bo_rng.uniform(_trans(lo, scale), _trans(hi, scale))
                    assign[k.name] = float(_inv_trans(tv, scale))
                else:
                    assign[k.name] = float(bo_rng.uniform(lo, hi))
            pool_assigns.append(assign)
            pool_vecs.append(_encode(assign))

        Xcand = np.asarray(pool_vecs, dtype=float)
        mu, sigma = _gp_posterior(X, y, Xcand, length=length, var=var, noise=noise)
        ei = _expected_improvement(mu, sigma, best=float(np.max(y)), xi=xi)
        pick = int(np.argmax(ei))
        assign_pick = dict(pool_assigns[pick])
        ov = json.loads(json.dumps(base_override))
        for path, val in assign_pick.items():
            _cfg_set_path(ov, path, float(val))
        base_tag = f"{family}__{tag or 'ours'}__bayes__t{t:02d}__{_stable_hash(assign_pick)}"
        row = _run_seed_aggregate(
            trials_csv=trials_csv,
            base_config_path=base_config_path,
            schedule_path=schedule_path,
            set_args=set_args,
            hpo_run_dir=hpo_run_dir,
            score_weights=score_weights,
            seed_stage="bayes",
            agg_stage="bayes.agg",
            base_tag=base_tag,
            override_seedless=ov,
            seeds=seeds,
            run_root=hpo_run_dir / "trial_runs" / family / (tag or "ours") / "bayes" / f"t{t:02d}",
        )
        proposals.append({"trial_tag": base_tag, "assign": assign_pick, "ei": float(ei[pick])})
        X = np.vstack([X, np.asarray(_encode(assign_pick), dtype=float)])
        y = np.append(y, float(row.get("score", "-inf")))
        if float(row.get("score", "-inf")) > float(best_row.get("score", "-inf")):
            best_row = row
            best_override = ov

    return {
        "enabled": True,
        "best_row": best_row,
        "best_override": best_override,
        "proposals": proposals,
        "configs": int(trials_budget),
    }


def run_hpo(cfg: Dict[str, Any], base_config_path: str, schedule_path: Optional[str], set_args: List[str]) -> None:
    _assert_capacity_unified(cfg)

    hpo_run_dir = _ensure_hpo_run_dir(cfg)
    trials_csv = hpo_run_dir / "trials.csv"
    best_path = hpo_run_dir / "best_hparams.json"
    plan_path = hpo_run_dir / "grid_plan.json"
    status_path = hpo_run_dir / "plan_status.json"
    bundle_path = hpo_run_dir / "hpo_bundle.json"
    best_curves_dir = hpo_run_dir / "best_curves"
    snapshot_path = hpo_run_dir / "config_snapshot.json"

    # provenance: keep exact configs used for this HPO run
    try:
        used_dir = hpo_run_dir / "configs_used"
        used_dir.mkdir(parents=True, exist_ok=True)
        bp = Path(str(base_config_path))
        if bp.exists() and bp.is_file():
            shutil.copy2(bp, used_dir / bp.name)
        if schedule_path:
            sp = Path(str(schedule_path))
            if sp.exists() and sp.is_file():
                shutil.copy2(sp, used_dir / sp.name)
        dump_json(
            hpo_run_dir / "config_provenance.json",
            {
                "base_config_path": str(base_config_path),
                "schedule_path": str(schedule_path) if schedule_path else None,
                "cli_set_args": list(set_args or []),
            },
        )
    except Exception:
        pass

    _atomic_write_json(snapshot_path, cfg)
    (hpo_run_dir / "base_merged.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    dump_json(hpo_run_dir / "cli_overrides.json", {"set_args": list(set_args or [])})

    hpo_cfg = cfg["hpo"]
    bandit = hpo_cfg["bandit"]
    score_cfg = bandit["score"]
    w_max = float(score_cfg["w_max"])
    w_final = float(score_cfg["w_final"])
    w_avg = float(score_cfg["w_avg"])
    score_weights = (w_max, w_final, w_avg)

    refine_seeds = [int(x) for x in bandit["refine_seeds"]]
    if len(refine_seeds) < 2:
        raise RuntimeError("[HPO] hpo.bandit.refine_seeds must have at least 2 seeds")
    refine_seeds = refine_seeds[:2]
    tags = _baseline_tags(cfg)

    grid_cfg = hpo_cfg["grid"]
    coord_cfg = hpo_cfg["coord"]
    coord_epochs = int(grid_cfg["baseline_search_epochs"])
    coord_max_steps_cfg = int(grid_cfg.get("baseline_search_max_steps", 0) or 0)
    rerank_cfg = grid_cfg["rerank"]
    top_k = int(coord_cfg["top_k"])
    refine_radix = int(coord_cfg["refine_radix"])
    refine_epochs = int(rerank_cfg["epochs"])
    refine_max_steps_cfg = int(rerank_cfg.get("max_steps", 0) or 0)
    bayes_trials = int(coord_cfg.get("bayes_trials", 48))

    task_cfg = cfg.get("task", {}) if isinstance(cfg.get("task", {}), dict) else {}
    multi_cfg = task_cfg.get("multi", {}) if isinstance(task_cfg.get("multi", {}), dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    train_multi_cfg = train_cfg.get("multi", {}) if isinstance(train_cfg.get("multi", {}), dict) else {}
    hpo_use_max_steps = bool(multi_cfg.get("enabled", False)) and (
        str(train_multi_cfg.get("steps_mode", "max_steps")).strip().lower() == "max_steps"
    )
    global_train_max_steps = int(train_cfg.get("max_steps", 0) or 0)

    def _resolve_stage_max_steps(stage_name: str, configured_steps: int) -> int:
        if not hpo_use_max_steps:
            return 0
        c = int(configured_steps)
        if c > 0:
            return c
        if global_train_max_steps > 0:
            return int(global_train_max_steps)
        raise RuntimeError(
            f"[HPO] {stage_name} requires positive max_steps in multi max_steps mode "
            "(set stage-specific *_max_steps or train.max_steps > 0)."
        )

    coord_max_steps = _resolve_stage_max_steps("coord_search", coord_max_steps_cfg)
    refine_max_steps = _resolve_stage_max_steps("local_refine", refine_max_steps_cfg)

    hpo_train_schedule = {
        "mode": "max_steps" if hpo_use_max_steps else "epochs",
        "global_train_max_steps": int(global_train_max_steps),
        "coord_search": {
            "epochs": int(coord_epochs),
            "max_steps": int(coord_max_steps),
            "configured_max_steps": int(coord_max_steps_cfg),
        },
        "local_refine": {
            "epochs": int(refine_epochs),
            "max_steps": int(refine_max_steps),
            "configured_max_steps": int(refine_max_steps_cfg),
        },
        "bayes": {
            "epochs": int(refine_epochs),
            "max_steps": int(refine_max_steps),
            "trials": int(bayes_trials),
        },
    }
    print(f"[HPO][SCHEDULE] mode={hpo_train_schedule['mode']} schedule={hpo_train_schedule}")
    total_trials = int(hpo_cfg["budget"]["total_trials"])
    family_budget = max(1, total_trials)
    baseline_tag_budget = _half_up_div(family_budget, len(tags))
    cagrad_tag_budget = _half_up_div(family_budget, len(tags))
    rng = np.random.default_rng(int(grid_cfg["rng_seed"]))

    chosen_ours_alpha: Optional[float] = float(_baseline_lora_from_method(cfg, "baseline_R")["alpha"])
    coord_plan = {
        "budget": {
            "total_trials_per_family": int(family_budget),
            "baseline_tag_budget": int(baseline_tag_budget),
            "cagrad_tag_budget": int(cagrad_tag_budget),
            "shared_order": list(coord_cfg["shared_order"]),
            "method_orders": json.loads(json.dumps(coord_cfg["method_orders"])),
            "top_k": int(top_k),
            "refine_radix": int(refine_radix),
            "bayes_trials": int(bayes_trials),
        },
        "hpo_train_schedule": hpo_train_schedule,
    }
    _atomic_write_json(plan_path, coord_plan)
    _atomic_write_json(status_path, {"status": "running", "updated_at": int(time.time())})

    baseline_results: Dict[str, Dict[str, Any]] = {}
    baseline_refine_results: Dict[str, Dict[str, Any]] = {}
    for tag in tags:
        baseline_results[tag] = _run_coordinate_descent(
            cfg=cfg,
            base_config_path=base_config_path,
            schedule_path=schedule_path,
            set_args=set_args,
            hpo_run_dir=hpo_run_dir,
            trials_csv=trials_csv,
            score_weights=score_weights,
            rng=rng,
            family="baseline",
            tag=tag,
            budget_trials=baseline_tag_budget,
            seeds=refine_seeds,
            stage_epochs=coord_epochs,
            stage_max_steps=coord_max_steps,
        )
        baseline_refine_results[tag] = _run_local_refine(
            cfg=cfg,
            base_config_path=base_config_path,
            schedule_path=schedule_path,
            set_args=set_args,
            hpo_run_dir=hpo_run_dir,
            trials_csv=trials_csv,
            score_weights=score_weights,
            family="baseline",
            tag=tag,
            coord_result=baseline_results[tag],
            seeds=refine_seeds,
            stage_epochs=refine_epochs,
            stage_max_steps=refine_max_steps,
            top_k=top_k,
            radix=refine_radix,
        )
        if float(baseline_refine_results[tag]["best_row"].get("score", "-inf")) >= float(baseline_results[tag]["best_row"].get("score", "-inf")):
            baseline_results[tag]["best_row"] = baseline_refine_results[tag]["best_row"]
            baseline_results[tag]["final_override"] = baseline_refine_results[tag]["best_override"]

    cagrad_results: Dict[str, Dict[str, Any]] = {}
    cagrad_refine_results: Dict[str, Dict[str, Any]] = {}
    for tag in tags:
        cagrad_results[tag] = _run_coordinate_descent(
            cfg=cfg,
            base_config_path=base_config_path,
            schedule_path=schedule_path,
            set_args=set_args,
            hpo_run_dir=hpo_run_dir,
            trials_csv=trials_csv,
            score_weights=score_weights,
            rng=rng,
            family="cagrad",
            tag=tag,
            budget_trials=cagrad_tag_budget,
            seeds=refine_seeds,
            stage_epochs=coord_epochs,
            stage_max_steps=coord_max_steps,
        )
        cagrad_refine_results[tag] = _run_local_refine(
            cfg=cfg,
            base_config_path=base_config_path,
            schedule_path=schedule_path,
            set_args=set_args,
            hpo_run_dir=hpo_run_dir,
            trials_csv=trials_csv,
            score_weights=score_weights,
            family="cagrad",
            tag=tag,
            coord_result=cagrad_results[tag],
            seeds=refine_seeds,
            stage_epochs=refine_epochs,
            stage_max_steps=refine_max_steps,
            top_k=top_k,
            radix=refine_radix,
        )
        if float(cagrad_refine_results[tag]["best_row"].get("score", "-inf")) >= float(cagrad_results[tag]["best_row"].get("score", "-inf")):
            cagrad_results[tag]["best_row"] = cagrad_refine_results[tag]["best_row"]
            cagrad_results[tag]["final_override"] = cagrad_refine_results[tag]["best_override"]

    ours_coord = _run_coordinate_descent(
        cfg=cfg,
        base_config_path=base_config_path,
        schedule_path=schedule_path,
        set_args=set_args,
        hpo_run_dir=hpo_run_dir,
        trials_csv=trials_csv,
        score_weights=score_weights,
        rng=rng,
        family="ours",
        tag=None,
        budget_trials=family_budget,
        seeds=refine_seeds,
        stage_epochs=coord_epochs,
        stage_max_steps=coord_max_steps,
    )
    ours_refine = _run_local_refine(
        cfg=cfg,
        base_config_path=base_config_path,
        schedule_path=schedule_path,
        set_args=set_args,
        hpo_run_dir=hpo_run_dir,
        trials_csv=trials_csv,
        score_weights=score_weights,
        family="ours",
        tag=None,
        coord_result=ours_coord,
        seeds=refine_seeds,
        stage_epochs=refine_epochs,
        stage_max_steps=refine_max_steps,
        top_k=top_k,
        radix=refine_radix,
    )
    ours_bayes = _run_bayes_after_refine(
        cfg=cfg,
        base_config_path=base_config_path,
        schedule_path=schedule_path,
        set_args=set_args,
        hpo_run_dir=hpo_run_dir,
        trials_csv=trials_csv,
        score_weights=score_weights,
        family="ours",
        tag=None,
        coord_result=ours_coord,
        refine_result=ours_refine,
        seeds=refine_seeds,
        stage_epochs=refine_epochs,
        stage_max_steps=refine_max_steps,
        trials_budget=bayes_trials,
    )

    best_overall = ours_coord["best_row"]
    best_source = "coord_search"
    if float(ours_refine["best_row"].get("score", "-inf")) >= float(best_overall.get("score", "-inf")):
        best_overall = ours_refine["best_row"]
        best_source = "local_refine"
    if ours_bayes.get("enabled", False) and float(ours_bayes["best_row"].get("score", "-inf")) >= float(best_overall.get("score", "-inf")):
        best_overall = ours_bayes["best_row"]
        best_source = "bayes"

    best_curves_dir.mkdir(parents=True, exist_ok=True)
    for tag, res in baseline_results.items():
        _copy_metrics_csv(res["best_row"], best_curves_dir / f"best_{tag}.csv")
    for tag, res in cagrad_results.items():
        _copy_metrics_csv(res["best_row"], best_curves_dir / f"best_baseline_cagrad_{tag}.csv")
    _copy_metrics_csv(best_overall, best_curves_dir / "best_ours.csv")

    if best_overall:
        baseline_best_lr_refined = {
            tag: float(_cfg_get_path(res["final_override"], "train.lr"))
            for tag, res in baseline_results.items()
        }
        baseline_cagrad_best_by_tag = {
            tag: {
                "lr": float(_cfg_get_path(res["final_override"], "train.lr")),
                "c": float(_cfg_get_path(res["final_override"], f"method.{tag}.cagrad.c")),
            }
            for tag, res in cagrad_results.items()
        }
        baseline_best_cfg_by_tag = {
            tag: json.loads(json.dumps(res["final_override"]))
            for tag, res in baseline_results.items()
        }
        baseline_cagrad_best_cfg_by_tag = {
            tag: json.loads(json.dumps(res["final_override"]))
            for tag, res in cagrad_results.items()
        }

        best_obj = {
            "best": best_overall,
            "best_source": best_source,
            "weights": {"w_max": w_max, "w_final": w_final, "w_avg": w_avg},
            "plan_path": str(plan_path),
            "status_path": str(status_path),
            "hpo_dir": str(hpo_run_dir),
            "baseline_best_lr_refined": baseline_best_lr_refined,
            "baseline_cagrad_best_by_tag": baseline_cagrad_best_by_tag,
            "baseline_best_cfg_by_tag": baseline_best_cfg_by_tag,
            "baseline_cagrad_best_cfg_by_tag": baseline_cagrad_best_cfg_by_tag,
            "hpo_train_schedule": hpo_train_schedule,
            "chosen_ours_alpha": chosen_ours_alpha,
            "coord_report": {
                "baseline": {tag: {"points_per_param": int(res["points_per_param"]), "sweeps": res["sweeps"]} for tag, res in baseline_results.items()},
                "cagrad": {tag: {"points_per_param": int(res["points_per_param"]), "sweeps": res["sweeps"]} for tag, res in cagrad_results.items()},
                "ours": {"points_per_param": int(ours_coord["points_per_param"]), "sweeps": ours_coord["sweeps"]},
            },
            "local_refine_report": {
                "baseline": baseline_refine_results,
                "cagrad": cagrad_refine_results,
                "ours": ours_refine,
            },
            "use_bayes": bool(cfg["hpo"]["use_bayes"]),
            "bayes_report": ours_bayes,
        }
        _atomic_write_json(best_path, best_obj)
        print(f"[HPO][OK] best saved: {best_path}")

        bundle = {
            "hpo_dir": str(hpo_run_dir),
            "best_path": str(best_path),
            "plan_path": str(plan_path),
            "status_path": str(status_path),
            "trials_csv": str(trials_csv),
            "best_curves_dir": str(best_curves_dir),
            "refine_seeds": refine_seeds,
            "baseline_tags": tags,
            "baseline_lora": {t: _baseline_lora_from_method(cfg, t) for t in tags},
            "baseline_best_lr_refined": baseline_best_lr_refined,
            "baseline_cagrad_best_by_tag": baseline_cagrad_best_by_tag,
            "baseline_best_cfg_by_tag": baseline_best_cfg_by_tag,
            "baseline_cagrad_best_cfg_by_tag": baseline_cagrad_best_cfg_by_tag,
            "hpo_train_schedule": hpo_train_schedule,
            "chosen_ours_alpha": chosen_ours_alpha,
            "coord_report": best_obj["coord_report"],
            "local_refine_report": best_obj["local_refine_report"],
            "use_bayes": bool(cfg["hpo"]["use_bayes"]),
            "bayes_report": ours_bayes,
            "config_snapshot": str(snapshot_path),
            "config_resolved": cfg,
        }
        _atomic_write_json(bundle_path, bundle)
        print(f"[HPO][OK] bundle saved: {bundle_path}")
        _atomic_write_json(status_path, {"status": "done", "updated_at": int(time.time())})
    else:
        print("[HPO][WARN] No aggregate rows found; cannot choose best.")
        print(f"[HPO][INFO] trials.csv: {trials_csv}")
