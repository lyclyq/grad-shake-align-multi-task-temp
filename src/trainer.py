# /home/lyclyq/Optimization/grad-shake-align/src/trainer.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Deque, Optional, Iterator
from collections import deque
import copy
from contextlib import contextmanager, nullcontext
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .loggingx import RunLogger
from .shake_align import ShakeAlignController, BlockStats
from .lora_layers import debug_check_dualrank_init


def _cfg_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _diag_metrics_from_info(info: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not info:
        return {}
    return {
        "diag/info_retention_ratio": float(info.get("info_retention_ratio", 0.0)),
        "diag/residual_visibility": float(info.get("residual_visibility", 0.0)),
        "diag/conflict_resolution_rate": float(info.get("conflict_resolution_rate", 0.0)),
    }


def _routing_metrics_from_info(info: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not info:
        return {}

    def _f(key: str) -> float:
        try:
            return float(info.get(key, float("nan")))
        except Exception:
            return float("nan")

    return {
        "train/expert_load_r": _f("expert_load_r"),
        "train/expert_load_R": _f("expert_load_R"),
        "train/load_cv": _f("load_cv"),
        "train/load_max_min_ratio": _f("load_max_min_ratio"),
        "train/utilization_ratio": _f("utilization_ratio"),
        "train/routing_entropy": _f("routing_entropy"),
        "train/routing_entropy_raw": _f("routing_entropy_raw"),
        "train/active_experts": _f("active_experts"),
        "train/expert_purity": _f("expert_purity"),
        "train/intra_expert_coherence": _f("intra_expert_coherence"),
        "train/intra_expert_conflict": _f("intra_expert_conflict"),
        "train/inter_expert_similarity": _f("inter_expert_similarity"),
    }


def _metric_token(name: str) -> str:
    raw = str(name).strip()
    if not raw:
        return "source"
    out = []
    prev_us = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        elif not prev_us:
            out.append("_")
            prev_us = True
    token = "".join(out).strip("_")
    return token or "source"


def _flatten_grad_vector_from_map(
    params: List[torch.nn.Parameter],
    grad_map: Dict[torch.nn.Parameter, torch.Tensor],
) -> torch.Tensor:
    if not params:
        return torch.zeros((0,), dtype=torch.float32)
    vecs: List[torch.Tensor] = []
    ref_device = None
    ref_dtype = torch.float32
    for p in params:
        g = grad_map.get(p, None)
        if g is not None:
            ref_device = g.device
            ref_dtype = g.dtype
            break
    if ref_device is None:
        ref_device = params[0].device
        ref_dtype = params[0].dtype
    for p in params:
        g = grad_map.get(p, None)
        if g is None:
            vecs.append(torch.zeros_like(p, device=ref_device, dtype=ref_dtype).flatten())
        else:
            vecs.append(g.detach().flatten())
    return torch.cat(vecs, dim=0)


def _coeff_var(vals: torch.Tensor) -> float:
    if vals.numel() <= 1:
        return float("nan")
    mean = float(vals.mean().item())
    if abs(mean) < 1e-12:
        return 0.0
    std = float(vals.std(unbiased=False).item())
    return float(std / (abs(mean) + 1e-12))


def _task_geometry_metrics(task_grads: Dict[str, torch.Tensor]) -> Dict[str, float]:
    if not task_grads:
        return {}

    names = list(task_grads.keys())
    vecs = [task_grads[n].detach().reshape(-1) for n in names]
    if not vecs:
        return {}

    mat = torch.stack(vecs, dim=0)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return {}

    norms = torch.norm(mat, dim=1, keepdim=True).clamp_min(1e-8)
    cos = (mat @ mat.t()) / (norms @ norms.t())

    out: Dict[str, float] = {}
    for idx, name in enumerate(names):
        tok = _metric_token(name)
        out[f"train/task_grad_norm/{tok}"] = float(norms[idx, 0].item())

    for i, ni in enumerate(names):
        ti = _metric_token(ni)
        for j, nj in enumerate(names):
            tj = _metric_token(nj)
            out[f"train/task_conflict_matrix/{ti}__{tj}"] = float(cos[i, j].item())

    if len(names) >= 2:
        mask = torch.triu(torch.ones_like(cos, dtype=torch.bool), diagonal=1)
        out["train/overall_conflict"] = float(cos[mask].mean().item()) if bool(mask.any()) else float("nan")
        out["train/gradient_norm_dispersion"] = _coeff_var(norms.squeeze(1))
    else:
        out["train/overall_conflict"] = float("nan")
        out["train/gradient_norm_dispersion"] = float("nan")

    return out


def _step_efficiency_metrics(step_start_t: float, device: torch.device) -> Dict[str, float]:
    step_time_ms = float((time.perf_counter() - float(step_start_t)) * 1000.0)
    peak_memory_mb = 0.0
    if device.type == "cuda":
        try:
            peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        except Exception:
            peak_memory_mb = 0.0
    return {
        "sys/time_per_step_ms": float(step_time_ms),
        "sys/peak_memory_mb": float(peak_memory_mb),
    }


def _loss_scalar(loss: Any) -> torch.Tensor:
    if not torch.is_tensor(loss):
        raise RuntimeError(f"[trainer] expected tensor loss, got {type(loss)}")
    return loss.mean() if loss.ndim > 0 else loss


def _batch_num_samples(batch: Dict[str, torch.Tensor]) -> int:
    for v in batch.values():
        if torch.is_tensor(v) and v.ndim > 0:
            return int(v.shape[0])
    raise RuntimeError("[trainer] could not infer batch size from batch payload")


def _slice_batch(batch: Dict[str, torch.Tensor], s: int, e: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.ndim > 0 and int(v.shape[0]) >= int(e):
            out[k] = v[s:e]
        else:
            out[k] = v
    return out


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _resolve_device_batch_size(cfg: Dict[str, Any], target_batch_size: int) -> int:
    raw = _as_int(_cfg_get(cfg, "train.device_batch_size", 0), 0)
    if raw <= 0:
        raw = int(target_batch_size)
    return max(1, min(int(raw), int(max(1, target_batch_size))))


def _iter_exec_spans(n: int, exec_batch_size: int) -> List[Tuple[int, int]]:
    chunk = max(1, int(exec_batch_size))
    return _split_indices(int(n), chunk, allow_tail=True)


def _forward_backward_mean_loss(
    model,
    batch_cpu: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    exec_batch_size: int,
    use_amp: bool,
) -> float:
    n = _batch_num_samples(batch_cpu)
    if n <= 0:
        raise RuntimeError("[trainer] empty batch in chunked backward")

    total_loss = 0.0
    for s, e in _iter_exec_spans(n, exec_batch_size):
        micro_cpu = _slice_batch(batch_cpu, s, e)
        micro = _move_batch_to_device(micro_cpu, device)
        weight = float(e - s) / float(n)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(use_amp)):
            out = model(**micro)
        micro_loss = _loss_scalar(out.loss)
        (micro_loss * weight).backward()
        total_loss += float(micro_loss.detach().item()) * weight
    return float(total_loss)


def _perf_cfg(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    tr = cfg.get("train", {}) or {}
    p = str(tr.get("precision", "fp32")).strip().lower()
    if p not in {"fp32", "bf16"}:
        raise RuntimeError(f"[trainer] train.precision must be one of fp32/bf16, got: {p!r}")
    use_amp = (p == "bf16") and (device.type == "cuda")
    tf32 = _as_bool(tr.get("tf32", True), True)
    compile_model = _as_bool(tr.get("compile", True), True)
    fused_adamw = _as_bool(tr.get("fused_adamw", True), True)
    return {
        "precision": p,
        "use_amp": bool(use_amp),
        "tf32": bool(tf32),
        "compile": bool(compile_model),
        "fused_adamw": bool(fused_adamw),
    }


def _setup_cuda_perf(perf: Dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    if bool(perf.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _maybe_compile_model(model, perf: Dict[str, Any], device: torch.device):
    if device.type != "cuda" or not bool(perf.get("compile", True)):
        return model
    if not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"[WARN] torch.compile disabled due to: {type(e).__name__}: {e}")
        return model


def _build_adamw(params, *, lr: float, weight_decay: float, perf: Dict[str, Any], device: torch.device):
    if device.type == "cuda" and bool(perf.get("fused_adamw", True)):
        try:
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _named_dualrank_lora_modules(model) -> Dict[str, torch.nn.Module]:
    out: Dict[str, torch.nn.Module] = {}
    for name, m in model.named_modules():
        if hasattr(m, "lora_A_r") and hasattr(m, "lora_A_hi"):
            out[name] = m
    return out


def _flatten_branch_grads(mod: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return scale-invariant gradient vectors (votes):
      g_r = concat(vec(dA_r), vec(dB_r)) / scaling_r
      g_hi = concat(vec(dA_hi), vec(dB_hi)) / scaling_hi

    IMPORTANT:
      - This is the ONLY place outside ShakeAlign that touches scaling,
        and it only DIVIDES to remove scaling for decision-making.
      - Writing scaled grads back happens ONLY inside ShakeAlign.
    """
    device = next(mod.parameters()).device

    if not hasattr(mod, "scaling_r") or not hasattr(mod, "scaling_hi"):
        raise RuntimeError(
            f"[ScalingMissing] {type(mod).__name__} missing scaling_r/scaling_hi. Module={mod}"
        )

    sr = float(getattr(mod, "scaling_r"))
    shi = float(getattr(mod, "scaling_hi"))
    if abs(sr) < 1e-12 or abs(shi) < 1e-12:
        raise RuntimeError(f"[ScalingInvalid] sr={sr} shi={shi} (must be nonzero)")

    def g_or_zeros(p: torch.nn.Parameter, s: float) -> torch.Tensor:
        if p.grad is None:
            return torch.zeros_like(p, device=device).flatten()
        g = p.grad.detach()
        g = g / s
        return g.flatten()

    g_r = torch.cat(
        [g_or_zeros(mod.lora_A_r, sr), g_or_zeros(mod.lora_B_r, sr)],
        dim=0,
    )
    g_hi = torch.cat(
        [g_or_zeros(mod.lora_A_hi, shi), g_or_zeros(mod.lora_B_hi, shi)],
        dim=0,
    )
    return g_r, g_hi


@contextmanager
def _set_dualrank_use_hi(model, use_hi: bool):
    touched = []
    for m in model.modules():
        if hasattr(m, "use_hi") and hasattr(m, "lora_A_hi") and hasattr(m, "lora_B_hi"):
            touched.append((m, bool(getattr(m, "use_hi"))))
            setattr(m, "use_hi", bool(use_hi))
    try:
        yield
    finally:
        for m, old in touched:
            setattr(m, "use_hi", old)


@torch.no_grad()
def evaluate_acc(
    model,
    loader: Any,
    device: torch.device,
    max_batches: int = 0,
    *,
    use_hi: Optional[bool] = None,
    device_batch_size: Optional[int] = None,
) -> float:
    if isinstance(loader, dict):
        vals = [
            evaluate_acc(model, ld, device, max_batches=max_batches, use_hi=use_hi, device_batch_size=device_batch_size)
            for ld in loader.values()
        ]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    was_training = bool(model.training)
    model.eval()
    correct = 0
    total = 0

    ctx = nullcontext() if use_hi is None else _set_dualrank_use_hi(model, use_hi)

    with ctx:
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            batch_n = _batch_num_samples(batch)
            exec_bs = _resolve_device_batch_size({"train": {"device_batch_size": device_batch_size or 0}}, batch_n)
            for s, e in _iter_exec_spans(batch_n, exec_bs):
                micro_cpu = _slice_batch(batch, s, e)
                micro = _move_batch_to_device(micro_cpu, device)
                out = model(**micro)
                preds = out.logits.argmax(dim=-1)
                labels = micro["labels"]
                correct += (preds == labels).sum().item()
                total += labels.numel()

    model.train(was_training)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_metrics(
    model,
    loader: Any,
    device: torch.device,
    max_batches: int = 0,
    *,
    use_hi: Optional[bool] = None,
    device_batch_size: Optional[int] = None,
) -> Dict[str, float]:
    if isinstance(loader, dict):
        vals = [
            evaluate_metrics(model, ld, device, max_batches=max_batches, use_hi=use_hi, device_batch_size=device_batch_size)
            for ld in loader.values()
        ]
        if not vals:
            return {"acc": 0.0, "loss": 0.0}
        return {
            "acc": float(sum(v["acc"] for v in vals) / len(vals)),
            "loss": float(sum(v["loss"] for v in vals) / len(vals)),
        }

    was_training = bool(model.training)
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    ctx = nullcontext() if use_hi is None else _set_dualrank_use_hi(model, use_hi)

    with ctx:
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            batch_n = _batch_num_samples(batch)
            exec_bs = _resolve_device_batch_size({"train": {"device_batch_size": device_batch_size or 0}}, batch_n)
            batch_loss = 0.0
            for s, e in _iter_exec_spans(batch_n, exec_bs):
                micro_cpu = _slice_batch(batch, s, e)
                micro = _move_batch_to_device(micro_cpu, device)
                out = model(**micro)
                preds = out.logits.argmax(dim=-1)
                labels = micro["labels"]
                weight = float(e - s) / float(batch_n)
                correct += (preds == labels).sum().item()
                total += labels.numel()
                batch_loss += float(_loss_scalar(out.loss).detach().item()) * weight
            total_loss += float(batch_loss)
            n_batches += 1

    model.train(was_training)
    return {
        "acc": float(correct / max(total, 1)),
        "loss": float(total_loss / max(n_batches, 1)),
    }


def _dbg_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("debug", {}) or {}
    return {
        "enabled": bool(d.get("enabled", False)),
        "print_every_steps": int(d.get("print_every_steps", 50)),
        "max_blocks_to_print": int(d.get("max_blocks_to_print", 3)),
        "dump_init": bool(d.get("dump_init", True)),
        "dump_votes": bool(d.get("dump_votes", True)),
        "dump_gates": bool(d.get("dump_gates", True)),
        "dump_grad_norms": bool(d.get("dump_grad_norms", True)),
        "dump_history": bool(d.get("dump_history", False)),
        "assert_hi_zero_init": bool(d.get("assert_hi_zero_init", True)),
    }


def _vote_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ours = cfg.get("method", {}).get("ours", {}) or {}
    voting = ours.get("voting", {}) or {}
    return {
        "samples_per_vote": int(voting.get("samples_per_vote", 4)),
        "keep_single_votes": bool(voting.get("keep_single_votes", True)),
        "allow_tail": bool(voting.get("allow_tail", True)),
    }


def _vote_history_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ours = cfg.get("method", {}).get("ours", {}) or {}
    hist = ours.get("history", {}) or {}
    return {
        "enabled": bool(hist.get("enabled", False)),
        "steps": int(hist.get("window_steps", 4)),
    }


def _split_indices(n: int, chunk: int, allow_tail: bool) -> List[Tuple[int, int]]:
    out = []
    s = 0
    while s < n:
        e = min(s + chunk, n)
        if (e - s) < chunk and (not allow_tail):
            break
        out.append((s, e))
        s = e
    return out


def _avg_last_k(seq: List[float], k: int) -> float:
    if not seq:
        return float("-inf")
    k = max(1, int(k))
    take = seq[-k:] if len(seq) >= k else seq
    return float(sum(take) / len(take))


def _iter_forever(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _baseline_solver_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    method = cfg.get("method", {}) or {}
    name = str(method.get("name", "")).strip()
    if name not in {"baseline_r", "baseline_R"}:
        return {"name": "avg", "c": 0.4, "eps": 1e-8, "inner_steps": 10, "inner_lr": 0.1}
    mcfg = method.get(name, {}) or {}
    solver = str(mcfg.get("grad_solver", "avg")).strip().lower()
    if solver not in {"avg", "cagrad"}:
        solver = "avg"
    cagrad_cfg = mcfg.get("cagrad", {}) or {}
    return {
        "name": solver,
        "c": float(cagrad_cfg.get("c", 0.4)),
        "eps": 1e-8,
        "inner_steps": 10,
        "inner_lr": 0.1,
    }


def _trainable_params(model) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flatten_full_grad_vector(params: List[torch.nn.Parameter]) -> torch.Tensor:
    if not params:
        return torch.zeros((0,), dtype=torch.float32)
    vecs: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            vecs.append(torch.zeros_like(p, memory_format=torch.contiguous_format).flatten())
        else:
            vecs.append(p.grad.detach().flatten())
    return torch.cat(vecs, dim=0)


def _assign_full_grad_vector(params: List[torch.nn.Parameter], vec: torch.Tensor) -> None:
    off = 0
    for p in params:
        n = int(p.numel())
        g = vec[off:off + n].view_as(p)
        off += n
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)


def _project_simplex(v: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection onto simplex {w >= 0, sum w = 1}.
    """
    if v.numel() == 0:
        return v
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    idx = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / idx > 0
    if not bool(torch.any(cond)):
        return torch.full_like(v, 1.0 / float(v.numel()))
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item()) + 1
    theta = float(cssv[rho - 1].item()) / float(rho)
    w = torch.clamp(v - theta, min=0.0)
    z = float(w.sum().item())
    if z <= 0:
        return torch.full_like(v, 1.0 / float(v.numel()))
    return w / z


def _cagrad_direction(
    grads: torch.Tensor,
    *,
    c: float,
    eps: float = 1e-8,
    inner_steps: int = 10,
    inner_lr: float = 0.1,
) -> torch.Tensor:
    """
    grads: [T, D], each row is one task gradient vector.
    returns: [D] conflict-averse direction.
    """
    if grads.ndim != 2:
        raise RuntimeError(f"[CAGrad] grads must be rank-2 [T, D], got shape={tuple(grads.shape)}")
    T = int(grads.shape[0])
    if T <= 0:
        raise RuntimeError("[CAGrad] empty grads")
    g0 = grads.mean(dim=0)
    if T == 1 or float(c) <= 0.0:
        return g0

    # Solve min_{w in simplex} 0.5 * w^T G w, where G is task-gram matrix.
    gram = grads @ grads.t()
    trace_g = torch.trace(gram)
    step_scale = float(inner_lr) / (float(trace_g.item()) + 1e-12)

    w = torch.full((T,), 1.0 / float(T), device=grads.device, dtype=grads.dtype)
    for _ in range(max(1, int(inner_steps))):
        grad_w = gram @ w
        w = _project_simplex(w - float(step_scale) * grad_w)

    s = (w.unsqueeze(1) * grads).sum(dim=0)
    g0_norm = torch.norm(g0)
    s_norm = torch.norm(s)
    scale = float(c) * g0_norm / (s_norm + float(eps))
    return g0 + scale * s


def train_one(
    cfg: dict,
    model,
    train_loader: Any,
    val_loader: Any,
    logger: RunLogger,
) -> Dict[str, float]:
    if _as_bool(_cfg_get(cfg, "task.multi.enabled", False), False):
        if not isinstance(train_loader, dict) or not isinstance(val_loader, dict):
            raise RuntimeError("[trainer] multi-task mode expects train_loader/val_loader to be dict(task->DataLoader)")
        return _train_one_multitask(cfg, model, train_loader, val_loader, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    perf = _perf_cfg(cfg, device)
    _setup_cuda_perf(perf, device)
    model = _maybe_compile_model(model, perf, device)
    _setup_cuda_perf(perf, device)
    model = _maybe_compile_model(model, perf, device)

    epochs = int(cfg["train"]["epochs"])
    effective_batch_size = int(cfg["train"]["batch_size"])
    resolved_device_batch_size = _resolve_device_batch_size(cfg, effective_batch_size)
    lr = float(cfg["train"]["lr"])
    warmup_ratio = float(cfg["train"]["warmup_ratio"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    single_steps_mode = str(_cfg_get(cfg, "train.single.steps_mode", "epochs")).strip().lower()
    if single_steps_mode not in {"epochs", "max_steps"}:
        raise RuntimeError(f"[trainer] train.single.steps_mode must be one of epochs/max_steps, got {single_steps_mode!r}")
    single_max_steps = _as_int(_cfg_get(cfg, "train.max_steps", 0), 0) if single_steps_mode == "max_steps" else 0
    if single_steps_mode == "max_steps" and single_max_steps <= 0:
        raise RuntimeError("[trainer] single-task max_steps mode requires train.max_steps > 0")

    opt = _build_adamw(model.parameters(), lr=lr, weight_decay=weight_decay, perf=perf, device=device)

    total_steps = int(single_max_steps) if single_steps_mode == "max_steps" else int(epochs * len(train_loader))
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    lora_modules = _named_dualrank_lora_modules(model)

    dbg = _dbg_cfg(cfg)
    vote_cfg = _vote_cfg(cfg)
    vh_cfg = _vote_history_cfg(cfg)

    is_ours = (cfg.get("method", {}).get("name", "") == "ours")

    # -------- eval config --------
    stage = str(cfg.get("stage", "") or "").strip().lower()

    eval_strategy = str(_cfg_get(cfg, "train.eval.strategy", "per_epoch")).strip().lower()
    if eval_strategy == "epoch":
        eval_strategy = "per_epoch"

    if stage != "final" and eval_strategy == "dense_early":
        eval_strategy = "per_epoch"

    dense_eval_per_epoch = _as_int(_cfg_get(cfg, "train.eval.dense_early_per_epoch", 8), 8)
    dense_early_epochs = _as_int(_cfg_get(cfg, "train.eval.dense_early_epochs", 2), 2)

    eval_every_steps = _as_int(_cfg_get(cfg, "train.eval.every_steps", 50), 50)
    eval_first_step = _as_bool(_cfg_get(cfg, "train.eval.first_step", False), False)
    eval_max_batches = _as_int(_cfg_get(cfg, "train.eval.max_batches", 0), 0)

    compute_train_acc = _as_bool(_cfg_get(cfg, "train.eval.compute_train_acc", True), True)
    train_acc_max_batches = _as_int(_cfg_get(cfg, "train.eval.train_max_batches", 0), 0)

    eval_r_only = _as_bool(_cfg_get(cfg, "train.eval.log_r_only", True), True)

    if eval_strategy not in {"dense_early", "per_epoch", "steps", "none"}:
        print(f"[WARN] Unknown train.eval.strategy={eval_strategy!r}, fallback to per_epoch")
        eval_strategy = "per_epoch"

    def _should_eval(
        *,
        global_step: int,
        step_in_epoch: int,
        steps_in_epoch: int,
        is_epoch_end: bool,
        ep: int,
    ) -> bool:
        if eval_strategy == "none":
            return False

        if is_epoch_end:
            return True

        if eval_strategy == "per_epoch":
            if eval_first_step and ep == 1 and step_in_epoch == 1:
                return True
            return False

        if eval_strategy == "dense_early":
            if ep > dense_early_epochs:
                return False
            k = max(1, int(dense_eval_per_epoch))
            every = max(1, steps_in_epoch // k)
            if step_in_epoch >= steps_in_epoch:
                return False
            return (step_in_epoch % every) == 0

        if eval_first_step and global_step == 1:
            return True
        if eval_every_steps <= 0:
            return False
        return (global_step % eval_every_steps) == 0

    # -------- controller config: disable double smoothing --------
    cfg_ctrl = copy.deepcopy(cfg)
    cfg_ctrl.setdefault("method", {})
    cfg_ctrl["method"].setdefault("ours", {})
    cfg_ctrl["method"]["ours"]["ema_H"] = 1
    cfg_ctrl["method"]["ours"].setdefault("history", {})
    cfg_ctrl["method"]["ours"]["history"]["enabled"] = False

    controller = ShakeAlignController(cfg_ctrl) if is_ours else None
    if controller is not None:
        controller.set_lora_modules(lora_modules)

    if controller is not None and dbg["enabled"] and dbg["dump_init"]:
        debug_check_dualrank_init(
            model,
            assert_hi_zero=dbg["assert_hi_zero_init"],
            max_blocks_to_print=dbg["max_blocks_to_print"],
        )

    best_val = -1.0
    best_epoch = -1

    val_history_epoch: List[float] = []
    val_r_only_history_epoch: List[float] = []
    train_history_epoch: List[float] = []
    train_r_only_history_epoch: List[float] = []

    global_step = 0
    last_eval_global_step = -1
    latest_probe_metrics: Dict[str, float] = {}

    # -------- vote history buffers --------
    vote_hist_enabled = bool(vh_cfg["enabled"])
    vote_hist_steps = max(1, int(vh_cfg["steps"]))
    vote_hist_r: Dict[str, Deque[torch.Tensor]] = {n: deque(maxlen=vote_hist_steps) for n in lora_modules.keys()}
    vote_hist_hi: Dict[str, Deque[torch.Tensor]] = {n: deque(maxlen=vote_hist_steps) for n in lora_modules.keys()}

    def _do_eval_and_log(step: int, ep: int) -> None:
        nonlocal best_val, best_epoch, last_eval_global_step
        if step == last_eval_global_step:
            return

        val_full = evaluate_metrics(
            model,
            val_loader,
            device,
            max_batches=eval_max_batches,
            use_hi=True,
            device_batch_size=resolved_device_batch_size,
        )
        val_acc = float(val_full["acc"])
        payload: Dict[str, Any] = {
            "val/acc": float(val_acc),
            "val/loss": float(val_full["loss"]),
            "epoch": int(ep),
            "probe/is_eval": 1.0,
            "sys/effective_batch_size": float(effective_batch_size),
            "sys/device_batch_size": float(resolved_device_batch_size),
        }

        val_acc_r = None
        if is_ours and eval_r_only:
            val_r = evaluate_metrics(
                model,
                val_loader,
                device,
                max_batches=eval_max_batches,
                use_hi=False,
                device_batch_size=resolved_device_batch_size,
            )
            val_acc_r = float(val_r["acc"])
            payload["val/acc_r_only"] = float(val_acc_r)
            payload["val/loss_r_only"] = float(val_r["loss"])

        if latest_probe_metrics:
            payload.update(latest_probe_metrics)

        if compute_train_acc:
            tr_full = evaluate_metrics(
                model,
                train_loader,
                device,
                max_batches=train_acc_max_batches,
                use_hi=True,
                device_batch_size=resolved_device_batch_size,
            )
            tr_acc = float(tr_full["acc"])
            payload["train/acc"] = float(tr_acc)
            payload["train/loss_eval"] = float(tr_full["loss"])
            payload["gap/train_minus_val"] = float(tr_acc - val_acc)

            if is_ours and eval_r_only:
                tr_r = evaluate_metrics(
                    model,
                    train_loader,
                    device,
                    max_batches=train_acc_max_batches,
                    use_hi=False,
                    device_batch_size=resolved_device_batch_size,
                )
                tr_acc_r = float(tr_r["acc"])
                payload["train/acc_r_only"] = float(tr_acc_r)
                payload["train/loss_r_only_eval"] = float(tr_r["loss"])
                if val_acc_r is not None:
                    payload["gap_r_only/train_minus_val_r_only"] = float(tr_acc_r - float(val_acc_r))

        logger.log(step, payload)
        last_eval_global_step = step

        if val_acc > best_val:
            best_val = float(val_acc)
            best_epoch = int(ep)

    def _record_epoch_snapshot(ep: int) -> None:
        val_acc = evaluate_acc(
            model,
            val_loader,
            device,
            max_batches=eval_max_batches,
            use_hi=True,
            device_batch_size=resolved_device_batch_size,
        )
        val_history_epoch.append(float(val_acc))

        if is_ours and eval_r_only:
            val_acc_r = evaluate_acc(
                model,
                val_loader,
                device,
                max_batches=eval_max_batches,
                use_hi=False,
                device_batch_size=resolved_device_batch_size,
            )
            val_r_only_history_epoch.append(float(val_acc_r))

        if compute_train_acc:
            tr_acc = evaluate_acc(
                model,
                train_loader,
                device,
                max_batches=train_acc_max_batches,
                use_hi=True,
                device_batch_size=resolved_device_batch_size,
            )
            train_history_epoch.append(float(tr_acc))

            if is_ours and eval_r_only:
                tr_acc_r = evaluate_acc(
                    model,
                    train_loader,
                    device,
                    max_batches=train_acc_max_batches,
                    use_hi=False,
                    device_batch_size=resolved_device_batch_size,
                )
                train_r_only_history_epoch.append(float(tr_acc_r))

    def _run_single_step(batch: Dict[str, torch.Tensor], *, ep: int, step_in_epoch: int) -> float:
        nonlocal latest_probe_metrics
        step_start_t = time.perf_counter()
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        batch_n = _batch_num_samples(batch)
        step_exec_batch_size = _resolve_device_batch_size(cfg, batch_n)
        step_loss_val = 0.0
        step_probe_metrics: Dict[str, float] = {}
        step_payload: Dict[str, Any] = {
            "epoch": int(ep),
            "step_in_epoch": int(step_in_epoch),
            "probe/is_eval": 0.0,
            "sys/effective_batch_size": float(batch_n),
            "sys/device_batch_size": float(step_exec_batch_size),
        }

        if controller is None:
            step_loss_val = _forward_backward_mean_loss(
                model,
                batch,
                device,
                exec_batch_size=step_exec_batch_size,
                use_amp=bool(perf["use_amp"]),
            )
            step_payload["train/loss"] = float(step_loss_val)
        else:
            bs = int(batch_n)
            spv = int(vote_cfg["samples_per_vote"])
            allow_tail = bool(vote_cfg["allow_tail"])

            windows = _split_indices(bs, spv, allow_tail=allow_tail)
            if len(windows) == 0:
                windows = [(0, bs)]

            step_votes_r: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}
            step_votes_hi: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}

            total_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
            loss_mean_batch = 0.0

            for (s, e) in windows:
                sub = _slice_batch(batch, s, e)
                win_weight = float(e - s) / float(bs)
                if win_weight <= 0.0:
                    raise RuntimeError(f"[trainer] invalid window weight: s={s} e={e} bs={bs}")

                opt.zero_grad(set_to_none=True)
                win_loss_val = _forward_backward_mean_loss(
                    model,
                    sub,
                    device,
                    exec_batch_size=step_exec_batch_size,
                    use_amp=bool(perf["use_amp"]),
                )
                loss_mean_batch += float(win_loss_val) * win_weight

                for name, mod in lora_modules.items():
                    g_r, g_hi = _flatten_branch_grads(mod)
                    step_votes_r[name].append(g_r)
                    step_votes_hi[name].append(g_hi)

                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        if p not in total_grads:
                            total_grads[p] = p.grad.detach().clone() * win_weight
                        else:
                            total_grads[p].add_(p.grad.detach(), alpha=win_weight)

            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p, g in total_grads.items():
                    p.grad = g
            step_loss_val = float(loss_mean_batch)

            packed_step_r: Dict[str, torch.Tensor] = {}
            packed_step_hi: Dict[str, torch.Tensor] = {}
            for name in lora_modules.keys():
                if not step_votes_r[name]:
                    continue
                packed_step_r[name] = torch.stack(step_votes_r[name], dim=0)
                packed_step_hi[name] = torch.stack(step_votes_hi[name], dim=0)

            if vote_hist_enabled:
                for name in lora_modules.keys():
                    if name in packed_step_r:
                        vote_hist_r[name].append(packed_step_r[name].detach())
                        vote_hist_hi[name].append(packed_step_hi[name].detach())

            votes_r: Dict[str, torch.Tensor] = {}
            votes_hi: Dict[str, torch.Tensor] = {}
            for name in lora_modules.keys():
                if vote_hist_enabled:
                    if len(vote_hist_r[name]) == 0:
                        continue
                    votes_r[name] = torch.cat(list(vote_hist_r[name]), dim=0)
                    votes_hi[name] = torch.cat(list(vote_hist_hi[name]), dim=0)
                else:
                    if name not in packed_step_r:
                        continue
                    votes_r[name] = packed_step_r[name]
                    votes_hi[name] = packed_step_hi[name]

            stats: Dict[str, BlockStats] = {}
            vote_sums: Dict[str, Dict[str, torch.Tensor]] = {}
            single_vote_blocks = 0

            for name in lora_modules.keys():
                if name not in votes_r:
                    continue
                vr = votes_r[name]
                vhi = votes_hi[name]
                if vr.shape[0] < 2:
                    single_vote_blocks += 1
                    continue

                fresh = controller.compute_stats_from_votes(vr, vhi)
                smooth = controller.ema_update(name, fresh)
                stats[name] = smooth
                vote_sums[name] = {"votes_r": vr, "votes_hi": vhi}

            if dbg["enabled"] and dbg["dump_votes"] and (global_step % dbg["print_every_steps"] == 0):
                any_name = next(iter(votes_r.keys()), None)
                v_total = int(votes_r[any_name].shape[0]) if any_name else 0
                print(f"[DBG][Step={global_step}] vote_hist={vote_hist_enabled} H={vote_hist_steps} V_total≈{v_total}")
                if single_vote_blocks > 0:
                    print(f"[DBG][Step={global_step}] single-vote blocks skipped={single_vote_blocks}")

            info = controller.apply_in_place_corrections(
                lora_modules=lora_modules,
                stats=stats,
                vote_sums=vote_sums,
                debug=bool(dbg["enabled"] and dbg["dump_gates"]),
                grad_norm_trace=bool(dbg["enabled"] and dbg["dump_grad_norms"]),
                debug_history=bool(dbg["enabled"] and dbg["dump_history"]),
            )
            step_payload.update(
                {
                    "train/loss": float(step_loss_val),
                    "train/gate0_triggered_blocks": info.get("triggered_blocks", 0.0),
                    "train/gate0_considered_blocks": info.get("considered_blocks", 0.0),
                    "train/gate0_trigger_rate": info.get("gate0_trigger_rate", 0.0),
                    "train/pull_to_r_blocks": info.get("pull_to_r_blocks", 0.0),
                    "train/pull_to_R_blocks": info.get("pull_to_R_blocks", 0.0),
                    "train/pull_to_r_rate": info.get("pull_to_r_rate", 0.0),
                    "train/pull_to_R_rate": info.get("pull_to_R_rate", 0.0),
                    "train/alpha_pull_mean": info.get("alpha_pull_mean", 0.0),
                    "train/alpha_pull_to_r_mean": info.get("alpha_pull_to_r_mean", 0.0),
                    "train/alpha_pull_to_R_mean": info.get("alpha_pull_to_R_mean", 0.0),
                    "train/single_vote_skipped_blocks": float(single_vote_blocks),
                    "train/tau_N": info.get("tau_N", 0.0),
                    "train/tau_D": info.get("tau_D", 0.0),
                }
            )
            step_probe_metrics.update(_diag_metrics_from_info(info))
            step_probe_metrics.update(_routing_metrics_from_info(info))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        step_probe_metrics.update(_step_efficiency_metrics(step_start_t, device))
        step_payload.update(step_probe_metrics)
        logger.log(global_step, step_payload)
        latest_probe_metrics = dict(step_probe_metrics)
        return float(step_loss_val)

    if single_steps_mode == "max_steps":
        model.train()
        steps_in_epoch = max(1, int(math.ceil(float(single_max_steps) / float(max(1, epochs)))))
        train_iter = _iter_forever(train_loader)
        pbar = tqdm(range(1, single_max_steps + 1), desc=f"single-task steps 1..{single_max_steps}")
        for cur_step in pbar:
            global_step = int(cur_step)
            ep = min(epochs, int((global_step - 1) // steps_in_epoch) + 1)
            step_in_epoch = int((global_step - 1) % steps_in_epoch) + 1
            is_epoch_end = bool(step_in_epoch >= steps_in_epoch or global_step >= single_max_steps)
            step_loss_val = _run_single_step(next(train_iter), ep=ep, step_in_epoch=step_in_epoch)
            pbar.set_postfix({"loss": f"{float(step_loss_val):.4f}"})

            if _should_eval(
                global_step=global_step,
                step_in_epoch=step_in_epoch,
                steps_in_epoch=steps_in_epoch,
                is_epoch_end=False,
                ep=ep,
            ):
                _do_eval_and_log(global_step, ep)

            if is_epoch_end and _should_eval(
                global_step=global_step,
                step_in_epoch=steps_in_epoch,
                steps_in_epoch=steps_in_epoch,
                is_epoch_end=True,
                ep=ep,
            ):
                _do_eval_and_log(global_step, ep)
                _record_epoch_snapshot(ep)
    else:
        for ep in range(1, epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"epoch {ep}/{epochs}")
            steps_in_epoch = len(train_loader)

            for step_in_epoch, batch in enumerate(pbar, start=1):
                global_step += 1
                step_loss_val = _run_single_step(batch, ep=ep, step_in_epoch=step_in_epoch)
                pbar.set_postfix({"loss": f"{float(step_loss_val):.4f}"})

                if _should_eval(
                    global_step=global_step,
                    step_in_epoch=step_in_epoch,
                    steps_in_epoch=steps_in_epoch,
                    is_epoch_end=False,
                    ep=ep,
                ):
                    _do_eval_and_log(global_step, ep)

            if _should_eval(
                global_step=global_step,
                step_in_epoch=steps_in_epoch,
                steps_in_epoch=steps_in_epoch,
                is_epoch_end=True,
                ep=ep,
            ):
                _do_eval_and_log(global_step, ep)
                _record_epoch_snapshot(ep)

    # -------- summary metrics --------
    if len(val_history_epoch) == 0:
        val_max = float(best_val)
        val_final = float(best_val)
        val_avg_last3 = float(best_val)
    else:
        val_max = float(max(val_history_epoch))
        val_final = float(val_history_epoch[-1])
        val_avg_last3 = _avg_last_k(val_history_epoch, 3)

    out: Dict[str, float] = {
        "best_val_acc": float(best_val),
        "best_epoch": float(best_epoch),
        "val_max": float(val_max),
        "val_final": float(val_final),
        "val_avg": float(val_avg_last3),
        "val_avg_last3ep": float(val_avg_last3),
    }

    if is_ours and eval_r_only and len(val_r_only_history_epoch) > 0:
        out["val_r_only_max"] = float(max(val_r_only_history_epoch))
        out["val_r_only_final"] = float(val_r_only_history_epoch[-1])
        out["val_r_only_avg_last3ep"] = float(_avg_last_k(val_r_only_history_epoch, 3))

    if len(train_history_epoch) > 0:
        out["train_final"] = float(train_history_epoch[-1])
        out["train_max"] = float(max(train_history_epoch))

    if is_ours and eval_r_only and len(train_r_only_history_epoch) > 0:
        out["train_r_only_final"] = float(train_r_only_history_epoch[-1])
        out["train_r_only_max"] = float(max(train_r_only_history_epoch))

    return out


def _train_one_multitask(
    cfg: dict,
    model,
    train_loader: Dict[str, DataLoader],
    val_loader: Dict[str, DataLoader],
    logger: RunLogger,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    perf = _perf_cfg(cfg, device)

    epochs = int(cfg["train"]["epochs"])
    steps_mode = str(_cfg_get(cfg, "train.multi.steps_mode", "max_steps")).strip().lower()
    if steps_mode not in {"max_steps", "epochs"}:
        raise RuntimeError("[trainer] train.multi.steps_mode must be one of: max_steps / epochs")

    if len(train_loader) == 0:
        raise RuntimeError("[trainer] empty multi-task train_loader")

    largest_steps_per_epoch = 0
    for name, loader in train_loader.items():
        n = int(len(loader))
        if n <= 0:
            raise RuntimeError(f"[trainer] task={name} has empty train DataLoader (len=0)")
        if n > largest_steps_per_epoch:
            largest_steps_per_epoch = n

    if steps_mode == "max_steps":
        max_steps = int(_cfg_get(cfg, "train.max_steps", 0))
        if max_steps <= 0:
            raise RuntimeError("[trainer] multi-task max_steps mode requires train.max_steps > 0")
        virtual_steps_per_epoch = max(1, int(math.ceil(float(max_steps) / float(max(1, epochs)))))
    else:
        # epoch mode: 1 epoch == largest task's dataloader length.
        max_steps = int(max(1, epochs) * max(1, largest_steps_per_epoch))
        virtual_steps_per_epoch = int(max(1, largest_steps_per_epoch))

    effective_batch_size_per_source = int(cfg["train"]["batch_size"])
    resolved_device_batch_size = _resolve_device_batch_size(cfg, effective_batch_size_per_source)
    lr = float(cfg["train"]["lr"])
    warmup_ratio = float(cfg["train"]["warmup_ratio"])
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))

    opt = _build_adamw(model.parameters(), lr=lr, weight_decay=weight_decay, perf=perf, device=device)

    total_steps = max_steps
    warmup_steps = int(total_steps * warmup_ratio)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    lora_modules = _named_dualrank_lora_modules(model)
    baseline_solver = _baseline_solver_cfg(cfg)
    baseline_params = _trainable_params(model)

    dbg = _dbg_cfg(cfg)
    vote_cfg = _vote_cfg(cfg)
    vh_cfg = _vote_history_cfg(cfg)

    is_ours = (cfg.get("method", {}).get("name", "") == "ours")

    # -------- eval config --------
    stage = str(cfg.get("stage", "") or "").strip().lower()

    eval_strategy = str(_cfg_get(cfg, "train.eval.strategy", "per_epoch")).strip().lower()
    if eval_strategy == "epoch":
        eval_strategy = "per_epoch"

    if stage != "final" and eval_strategy == "dense_early":
        eval_strategy = "per_epoch"

    dense_eval_per_epoch = _as_int(_cfg_get(cfg, "train.eval.dense_early_per_epoch", 8), 8)
    dense_early_epochs = _as_int(_cfg_get(cfg, "train.eval.dense_early_epochs", 2), 2)

    eval_every_steps = _as_int(_cfg_get(cfg, "train.eval.every_steps", 50), 50)
    eval_first_step = _as_bool(_cfg_get(cfg, "train.eval.first_step", False), False)
    eval_max_batches = _as_int(_cfg_get(cfg, "train.eval.max_batches", 0), 0)

    compute_train_acc = _as_bool(_cfg_get(cfg, "train.eval.compute_train_acc", True), True)
    train_acc_max_batches = _as_int(_cfg_get(cfg, "train.eval.train_max_batches", 0), 0)

    eval_r_only = _as_bool(_cfg_get(cfg, "train.eval.log_r_only", True), True)

    if steps_mode == "max_steps" and eval_strategy == "per_epoch":
        # max_steps multi-task training has no natural dataset epoch boundary.
        eval_strategy = "steps"

    if eval_strategy not in {"dense_early", "per_epoch", "steps", "none"}:
        print(f"[WARN] Unknown train.eval.strategy={eval_strategy!r}, fallback to per_epoch")
        eval_strategy = "per_epoch"

    def _should_eval(
        *,
        global_step: int,
        step_in_epoch: int,
        steps_in_epoch: int,
        is_epoch_end: bool,
        ep: int,
    ) -> bool:
        if eval_strategy == "none":
            return False

        if is_epoch_end:
            return True

        if eval_strategy == "per_epoch":
            if eval_first_step and ep == 1 and step_in_epoch == 1:
                return True
            return False

        if eval_strategy == "dense_early":
            if ep > dense_early_epochs:
                return False
            k = max(1, int(dense_eval_per_epoch))
            every = max(1, steps_in_epoch // k)
            if step_in_epoch >= steps_in_epoch:
                return False
            return (step_in_epoch % every) == 0

        if eval_first_step and global_step == 1:
            return True
        if eval_every_steps <= 0:
            return False
        return (global_step % eval_every_steps) == 0

    # -------- controller config: disable double smoothing --------
    cfg_ctrl = copy.deepcopy(cfg)
    cfg_ctrl.setdefault("method", {})
    cfg_ctrl["method"].setdefault("ours", {})
    cfg_ctrl["method"]["ours"]["ema_H"] = 1
    cfg_ctrl["method"]["ours"].setdefault("history", {})
    cfg_ctrl["method"]["ours"]["history"]["enabled"] = False

    controller = ShakeAlignController(cfg_ctrl) if is_ours else None
    if controller is not None:
        controller.set_lora_modules(lora_modules)

    if controller is not None and dbg["enabled"] and dbg["dump_init"]:
        debug_check_dualrank_init(
            model,
            assert_hi_zero=dbg["assert_hi_zero_init"],
            max_blocks_to_print=dbg["max_blocks_to_print"],
        )

    best_val = -1.0
    best_epoch = -1

    val_history_epoch: List[float] = []
    val_r_only_history_epoch: List[float] = []
    train_history_epoch: List[float] = []
    train_r_only_history_epoch: List[float] = []

    global_step = 0
    last_eval_global_step = -1
    latest_probe_metrics: Dict[str, float] = {}

    # -------- vote history buffers --------
    vote_hist_enabled = bool(vh_cfg["enabled"])
    vote_hist_steps = max(1, int(vh_cfg["steps"]))
    vote_hist_r: Dict[str, Deque[torch.Tensor]] = {n: deque(maxlen=vote_hist_steps) for n in lora_modules.keys()}
    vote_hist_hi: Dict[str, Deque[torch.Tensor]] = {n: deque(maxlen=vote_hist_steps) for n in lora_modules.keys()}

    def _do_eval_and_log(step: int, ep: int) -> None:
        nonlocal best_val, best_epoch, last_eval_global_step
        if step == last_eval_global_step:
            return

        val_full = evaluate_metrics(
            model,
            val_loader,
            device,
            max_batches=eval_max_batches,
            use_hi=True,
            device_batch_size=resolved_device_batch_size,
        )
        val_acc = float(val_full["acc"])
        payload: Dict[str, Any] = {
            "val/acc": float(val_acc),
            "val/loss": float(val_full["loss"]),
            "epoch": int(ep),
            "probe/is_eval": 1.0,
            "sys/effective_batch_size_per_source": float(effective_batch_size_per_source),
            "sys/device_batch_size": float(resolved_device_batch_size),
        }

        val_acc_r = None
        if is_ours and eval_r_only:
            val_r = evaluate_metrics(
                model,
                val_loader,
                device,
                max_batches=eval_max_batches,
                use_hi=False,
                device_batch_size=resolved_device_batch_size,
            )
            val_acc_r = float(val_r["acc"])
            payload["val/acc_r_only"] = float(val_acc_r)
            payload["val/loss_r_only"] = float(val_r["loss"])

        if latest_probe_metrics:
            payload.update(latest_probe_metrics)

        if compute_train_acc:
            tr_full = evaluate_metrics(
                model,
                train_loader,
                device,
                max_batches=train_acc_max_batches,
                use_hi=True,
                device_batch_size=resolved_device_batch_size,
            )
            tr_acc = float(tr_full["acc"])
            payload["train/acc"] = float(tr_acc)
            payload["train/loss_eval"] = float(tr_full["loss"])
            payload["gap/train_minus_val"] = float(tr_acc - val_acc)

            if is_ours and eval_r_only:
                tr_r = evaluate_metrics(
                    model,
                    train_loader,
                    device,
                    max_batches=train_acc_max_batches,
                    use_hi=False,
                    device_batch_size=resolved_device_batch_size,
                )
                tr_acc_r = float(tr_r["acc"])
                payload["train/acc_r_only"] = float(tr_acc_r)
                payload["train/loss_r_only_eval"] = float(tr_r["loss"])
                if val_acc_r is not None:
                    payload["gap_r_only/train_minus_val_r_only"] = float(tr_acc_r - float(val_acc_r))

        logger.log(step, payload)
        last_eval_global_step = step

        if val_acc > best_val:
            best_val = float(val_acc)
            best_epoch = int(ep)

    train_iters: Dict[str, Iterator[Dict[str, torch.Tensor]]] = {
        name: _iter_forever(loader) for name, loader in train_loader.items()
    }

    pbar = tqdm(range(1, max_steps + 1), desc=f"multitask steps 1..{max_steps}")
    for cur_step in pbar:
        global_step = int(cur_step)
        ep = min(epochs, int((global_step - 1) // virtual_steps_per_epoch) + 1)
        step_in_epoch = int((global_step - 1) % virtual_steps_per_epoch) + 1
        is_epoch_end = bool(step_in_epoch >= virtual_steps_per_epoch or global_step >= max_steps)
        step_start_t = time.perf_counter()
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass

        batch_by_task: Dict[str, Dict[str, torch.Tensor]] = {}
        for task_name, it in train_iters.items():
            batch_by_task[task_name] = next(it)

        step_loss_val = 0.0
        step_probe_metrics: Dict[str, float] = {}
        step_payload: Dict[str, Any] = {
            "epoch": int(ep),
            "step_in_epoch": int(step_in_epoch),
            "probe/is_eval": 0.0,
            "sys/effective_batch_size_per_source": float(effective_batch_size_per_source),
            "sys/device_batch_size": float(resolved_device_batch_size),
        }

        if controller is None:
            grad_vecs: List[torch.Tensor] = []
            task_grad_map: Dict[str, torch.Tensor] = {}
            loss_sum = 0.0
            for task_name, batch in batch_by_task.items():
                opt.zero_grad(set_to_none=True)
                loss_val = _forward_backward_mean_loss(
                    model,
                    batch,
                    device,
                    exec_batch_size=_resolve_device_batch_size(cfg, _batch_num_samples(batch)),
                    use_amp=bool(perf["use_amp"]),
                )
                grad_vec = _flatten_full_grad_vector(baseline_params)
                grad_vecs.append(grad_vec)
                task_grad_map[task_name] = grad_vec
                loss_sum += float(loss_val)

            if grad_vecs:
                grads = torch.stack(grad_vecs, dim=0)
                if baseline_solver["name"] == "cagrad":
                    direction = _cagrad_direction(
                        grads,
                        c=float(baseline_solver["c"]),
                        eps=float(baseline_solver["eps"]),
                        inner_steps=int(baseline_solver["inner_steps"]),
                        inner_lr=float(baseline_solver["inner_lr"]),
                    )
                else:
                    direction = grads.mean(dim=0)
                opt.zero_grad(set_to_none=True)
                _assign_full_grad_vector(baseline_params, direction)

            step_loss_val = float(loss_sum / max(1, len(grad_vecs)))
            step_payload.update(
                {
                    "train/loss": float(step_loss_val),
                    "train/num_tasks": float(len(batch_by_task)),
                    "train/grad_solver_cagrad": 1.0 if baseline_solver["name"] == "cagrad" else 0.0,
                    "train/cagrad_c": float(baseline_solver["c"]) if baseline_solver["name"] == "cagrad" else 0.0,
                }
            )
            step_probe_metrics.update(_task_geometry_metrics(task_grad_map))
        else:
            spv = int(vote_cfg["samples_per_vote"])
            allow_tail = bool(vote_cfg["allow_tail"])

            total_samples = int(sum(_batch_num_samples(batch) for batch in batch_by_task.values()))
            if total_samples <= 0:
                raise RuntimeError("[trainer] empty total_samples in multi-task step")

            step_votes_r: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}
            step_votes_hi: Dict[str, List[torch.Tensor]] = {n: [] for n in lora_modules.keys()}

            total_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
            task_grad_maps: Dict[str, Dict[torch.nn.Parameter, torch.Tensor]] = {
                task_name: {} for task_name in batch_by_task.keys()
            }
            loss_mean_batch = 0.0

            for task_name, batch in batch_by_task.items():
                bs = int(_batch_num_samples(batch))
                windows = _split_indices(bs, spv, allow_tail=allow_tail)
                if len(windows) == 0:
                    windows = [(0, bs)]

                for (s, e) in windows:
                    sub = _slice_batch(batch, s, e)
                    win_weight = float(e - s) / float(total_samples)
                    task_win_weight = float(e - s) / float(bs)
                    if win_weight <= 0.0:
                        raise RuntimeError(f"[trainer] invalid window weight: s={s} e={e} total={total_samples}")
                    if task_win_weight <= 0.0:
                        raise RuntimeError(f"[trainer] invalid task window weight: s={s} e={e} bs={bs}")

                    opt.zero_grad(set_to_none=True)
                    win_loss_val = _forward_backward_mean_loss(
                        model,
                        sub,
                        device,
                        exec_batch_size=_resolve_device_batch_size(cfg, e - s),
                        use_amp=bool(perf["use_amp"]),
                    )
                    loss_mean_batch += float(win_loss_val) * win_weight

                    # scale-invariant votes (divide scaling here)
                    for name, mod in lora_modules.items():
                        g_r, g_hi = _flatten_branch_grads(mod)
                        step_votes_r[name].append(g_r)
                        step_votes_hi[name].append(g_hi)

                    with torch.no_grad():
                        for p in model.parameters():
                            if p.grad is None:
                                continue
                            if p not in total_grads:
                                total_grads[p] = p.grad.detach().clone() * win_weight
                            else:
                                total_grads[p].add_(p.grad.detach(), alpha=win_weight)
                            if p not in task_grad_maps[task_name]:
                                task_grad_maps[task_name][p] = p.grad.detach().clone() * task_win_weight
                            else:
                                task_grad_maps[task_name][p].add_(p.grad.detach(), alpha=task_win_weight)

            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p, g in total_grads.items():
                    p.grad = g
            step_loss_val = float(loss_mean_batch)
            task_grad_map = {
                task_name: _flatten_grad_vector_from_map(baseline_params, grad_map)
                for task_name, grad_map in task_grad_maps.items()
                if grad_map
            }

            packed_step_r: Dict[str, torch.Tensor] = {}
            packed_step_hi: Dict[str, torch.Tensor] = {}
            for name in lora_modules.keys():
                if not step_votes_r[name]:
                    continue
                packed_step_r[name] = torch.stack(step_votes_r[name], dim=0)
                packed_step_hi[name] = torch.stack(step_votes_hi[name], dim=0)

            if vote_hist_enabled:
                for name in lora_modules.keys():
                    if name in packed_step_r:
                        vote_hist_r[name].append(packed_step_r[name].detach())
                        vote_hist_hi[name].append(packed_step_hi[name].detach())

            votes_r: Dict[str, torch.Tensor] = {}
            votes_hi: Dict[str, torch.Tensor] = {}
            for name in lora_modules.keys():
                if vote_hist_enabled:
                    if len(vote_hist_r[name]) == 0:
                        continue
                    votes_r[name] = torch.cat(list(vote_hist_r[name]), dim=0)
                    votes_hi[name] = torch.cat(list(vote_hist_hi[name]), dim=0)
                else:
                    if name not in packed_step_r:
                        continue
                    votes_r[name] = packed_step_r[name]
                    votes_hi[name] = packed_step_hi[name]

            stats: Dict[str, BlockStats] = {}
            vote_sums: Dict[str, Dict[str, torch.Tensor]] = {}
            single_vote_blocks = 0

            for name in lora_modules.keys():
                if name not in votes_r:
                    continue
                vr = votes_r[name]
                vhi = votes_hi[name]
                if vr.shape[0] < 2:
                    single_vote_blocks += 1
                    continue

                fresh = controller.compute_stats_from_votes(vr, vhi)
                smooth = controller.ema_update(name, fresh)
                stats[name] = smooth
                vote_sums[name] = {"votes_r": vr, "votes_hi": vhi}

            if dbg["enabled"] and dbg["dump_votes"] and (global_step % dbg["print_every_steps"] == 0):
                any_name = next(iter(votes_r.keys()), None)
                v_total = int(votes_r[any_name].shape[0]) if any_name else 0
                print(f"[DBG][Step={global_step}] vote_hist={vote_hist_enabled} H={vote_hist_steps} V_total≈{v_total}")
                if single_vote_blocks > 0:
                    print(f"[DBG][Step={global_step}] single-vote blocks skipped={single_vote_blocks}")

            info = controller.apply_in_place_corrections(
                lora_modules=lora_modules,
                stats=stats,
                vote_sums=vote_sums,
                debug=bool(dbg["enabled"] and dbg["dump_gates"]),
                grad_norm_trace=bool(dbg["enabled"] and dbg["dump_grad_norms"]),
                debug_history=bool(dbg["enabled"] and dbg["dump_history"]),
            )
            step_payload.update(
                {
                    "train/loss": float(step_loss_val),
                    "train/num_tasks": float(len(batch_by_task)),
                    "train/gate0_triggered_blocks": info.get("triggered_blocks", 0.0),
                    "train/gate0_considered_blocks": info.get("considered_blocks", 0.0),
                    "train/gate0_trigger_rate": info.get("gate0_trigger_rate", 0.0),
                    "train/pull_to_r_blocks": info.get("pull_to_r_blocks", 0.0),
                    "train/pull_to_R_blocks": info.get("pull_to_R_blocks", 0.0),
                    "train/pull_to_r_rate": info.get("pull_to_r_rate", 0.0),
                    "train/pull_to_R_rate": info.get("pull_to_R_rate", 0.0),
                    "train/alpha_pull_mean": info.get("alpha_pull_mean", 0.0),
                    "train/alpha_pull_to_r_mean": info.get("alpha_pull_to_r_mean", 0.0),
                    "train/alpha_pull_to_R_mean": info.get("alpha_pull_to_R_mean", 0.0),
                    "train/single_vote_skipped_blocks": float(single_vote_blocks),
                    "train/tau_N": info.get("tau_N", 0.0),
                    "train/tau_D": info.get("tau_D", 0.0),
                }
            )
            step_probe_metrics.update(_task_geometry_metrics(task_grad_map))
            step_probe_metrics.update(_diag_metrics_from_info(info))
            step_probe_metrics.update(_routing_metrics_from_info(info))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        step_probe_metrics.update(_step_efficiency_metrics(step_start_t, device))
        step_payload.update(step_probe_metrics)
        logger.log(global_step, step_payload)
        latest_probe_metrics = dict(step_probe_metrics)

        pbar.set_postfix({"loss": f"{float(step_loss_val):.4f}"})

        if _should_eval(
            global_step=global_step,
            step_in_epoch=step_in_epoch,
            steps_in_epoch=virtual_steps_per_epoch,
            is_epoch_end=False,
            ep=ep,
        ):
            _do_eval_and_log(global_step, ep)

        if is_epoch_end and _should_eval(
            global_step=global_step,
            step_in_epoch=virtual_steps_per_epoch,
            steps_in_epoch=virtual_steps_per_epoch,
            is_epoch_end=True,
            ep=ep,
        ):
            _do_eval_and_log(global_step, ep)

            # epoch-end summary snapshots
            val_acc = evaluate_acc(
                model,
                val_loader,
                device,
                max_batches=eval_max_batches,
                use_hi=True,
                device_batch_size=resolved_device_batch_size,
            )
            val_history_epoch.append(float(val_acc))

            if is_ours and eval_r_only:
                val_acc_r = evaluate_acc(
                    model,
                    val_loader,
                    device,
                    max_batches=eval_max_batches,
                    use_hi=False,
                    device_batch_size=resolved_device_batch_size,
                )
                val_r_only_history_epoch.append(float(val_acc_r))

            if compute_train_acc:
                tr_acc = evaluate_acc(
                    model,
                    train_loader,
                    device,
                    max_batches=train_acc_max_batches,
                    use_hi=True,
                    device_batch_size=resolved_device_batch_size,
                )
                train_history_epoch.append(float(tr_acc))

                if is_ours and eval_r_only:
                    tr_acc_r = evaluate_acc(
                        model,
                        train_loader,
                        device,
                        max_batches=train_acc_max_batches,
                        use_hi=False,
                        device_batch_size=resolved_device_batch_size,
                    )
                    train_r_only_history_epoch.append(float(tr_acc_r))

    # -------- summary metrics --------
    if len(val_history_epoch) == 0:
        val_max = float(best_val)
        val_final = float(best_val)
        val_avg_last3 = float(best_val)
    else:
        val_max = float(max(val_history_epoch))
        val_final = float(val_history_epoch[-1])
        val_avg_last3 = _avg_last_k(val_history_epoch, 3)

    out: Dict[str, float] = {
        "best_val_acc": float(best_val),
        "best_epoch": float(best_epoch),
        "val_max": float(val_max),
        "val_final": float(val_final),
        "val_avg": float(val_avg_last3),
        "val_avg_last3ep": float(val_avg_last3),
    }

    if is_ours and eval_r_only and len(val_r_only_history_epoch) > 0:
        out["val_r_only_max"] = float(max(val_r_only_history_epoch))
        out["val_r_only_final"] = float(val_r_only_history_epoch[-1])
        out["val_r_only_avg_last3ep"] = float(_avg_last_k(val_r_only_history_epoch, 3))

    if len(train_history_epoch) > 0:
        out["train_final"] = float(train_history_epoch[-1])
        out["train_max"] = float(max(train_history_epoch))

    if is_ours and eval_r_only and len(train_r_only_history_epoch) > 0:
        out["train_r_only_final"] = float(train_r_only_history_epoch[-1])
        out["train_r_only_max"] = float(max(train_r_only_history_epoch))

    return out
