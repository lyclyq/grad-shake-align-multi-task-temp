# src/shake_align.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class BlockStats:
    C_r: float
    C_R: float
    A_b: float


def _mean_pairwise_cos(vectors: List[torch.Tensor], eps: float) -> float:
    if len(vectors) < 2:
        return float("nan")
    buckets: Dict[int, List[torch.Tensor]] = {}
    for v in vectors:
        flat = v.detach().reshape(-1)
        buckets.setdefault(int(flat.numel()), []).append(flat)

    weighted_sum = 0.0
    weighted_pairs = 0
    for group in buckets.values():
        if len(group) < 2:
            continue
        mat = torch.stack(group, dim=0)
        norms = torch.norm(mat, dim=1, keepdim=True).clamp_min(float(eps))
        cos = (mat @ mat.t()) / (norms @ norms.t())
        mask = torch.triu(torch.ones_like(cos, dtype=torch.bool), diagonal=1)
        if not bool(mask.any()):
            continue
        pair_count = int(mask.sum().item())
        weighted_sum += float(cos[mask].mean().item()) * float(pair_count)
        weighted_pairs += pair_count
    if weighted_pairs <= 0:
        return float("nan")
    return float(weighted_sum / float(weighted_pairs))


def _mean_cross_bucket_cos(vectors_a: List[torch.Tensor], vectors_b: List[torch.Tensor], eps: float) -> float:
    if not vectors_a or not vectors_b:
        return float("nan")

    buckets_a: Dict[int, List[torch.Tensor]] = {}
    buckets_b: Dict[int, List[torch.Tensor]] = {}
    for v in vectors_a:
        flat = v.detach().reshape(-1)
        buckets_a.setdefault(int(flat.numel()), []).append(flat)
    for v in vectors_b:
        flat = v.detach().reshape(-1)
        buckets_b.setdefault(int(flat.numel()), []).append(flat)

    weighted_sum = 0.0
    weighted_count = 0
    for dim in sorted(set(buckets_a.keys()) & set(buckets_b.keys())):
        mat_a = torch.stack(buckets_a[dim], dim=0)
        mat_b = torch.stack(buckets_b[dim], dim=0)
        mean_a = mat_a.mean(dim=0)
        mean_b = mat_b.mean(dim=0)
        denom = torch.norm(mean_a).clamp_min(float(eps)) * torch.norm(mean_b).clamp_min(float(eps))
        sim = float(torch.dot(mean_a, mean_b).item() / float(denom.item()))
        count = int(min(mat_a.shape[0], mat_b.shape[0]))
        weighted_sum += sim * float(count)
        weighted_count += count
    if weighted_count <= 0:
        return float("nan")
    return float(weighted_sum / float(weighted_count))


class ShakeAlignController:
    """
    Implements (paper-aligned):
      - vote stats: C_r, C_R, A_b
      - Gate0 trigger
      - chi* routing
      - pull on head (either R->r or r->R)
      - compensation:
          g_hi <- tail(g_R_exec) + P(head(g_R_exec) - g_r_exec)
    """

    def __init__(self, cfg: Dict[str, Any], lora_modules: Optional[Dict[str, torch.nn.Module]] = None):
        self.cfg = cfg
        self.eps = 1e-8

        ours = cfg.get("method", {}).get("ours", {}) or {}
        self.V = int(ours.get("votes", 4))

        # legacy EMA window
        self.H_legacy = int(ours.get("ema_H", 4))

        # history smoothing config
        hist = ours.get("history", {}) or {}
        self.hist_enabled = bool(hist.get("enabled", False))
        self.hist_window = int(hist.get("window_steps", self.H_legacy))
        self.hist_weighting = str(hist.get("weighting", "exp"))
        self.hist_beta = float(hist.get("exp_beta", 0.7))

        self._hist: Dict[str, Deque[BlockStats]] = {}
        self._ema: Dict[str, BlockStats] = {}

        self.lora_modules: Dict[str, torch.nn.Module] = lora_modules or {}

    def set_lora_modules(self, lora_modules: Dict[str, torch.nn.Module]) -> None:
        self.lora_modules = lora_modules

    # -------------------------
    # Stats
    # -------------------------
    def compute_stats_from_votes(self, votes_r: torch.Tensor, votes_hi: torch.Tensor) -> BlockStats:
        V = votes_r.shape[0]
        if V < 2:
            return BlockStats(C_r=0.0, C_R=0.0, A_b=1.0)

        # Gram matrices
        D_r = votes_r @ votes_r.t()
        if votes_hi is not None and votes_hi.numel():
            D_hi = votes_hi @ votes_hi.t()
        else:
            D_hi = torch.zeros_like(D_r)

        D_R = D_r + D_hi

        # diagonal energies
        S_r = torch.diag(D_r).clamp_min(0)
        S_R = torch.diag(D_R).clamp_min(0)

        # upper triangle mask (exclude diagonal)
        mask = torch.triu(torch.ones((V, V), device=votes_r.device, dtype=torch.bool), diagonal=1)

        denom_r = (S_r.sqrt().unsqueeze(1) * S_r.sqrt().unsqueeze(0) + self.eps)
        denom_R = (S_R.sqrt().unsqueeze(1) * S_R.sqrt().unsqueeze(0) + self.eps)

        C_r = (D_r[mask] / denom_r[mask]).mean().item()
        C_R = (D_R[mask] / denom_R[mask]).mean().item()

        sigma_r_sq = D_r.sum()
        sigma_hi_sq = (
            D_hi.sum()
            if (votes_hi is not None and votes_hi.numel())
            else torch.tensor(0.0, device=votes_r.device)
        )
        sigma_R_sq = sigma_r_sq + sigma_hi_sq

        A_b = (sigma_r_sq.sqrt() / (sigma_R_sq.sqrt() + self.eps)).item()

        return BlockStats(C_r=float(C_r), C_R=float(C_R), A_b=float(A_b))

    # -------------------------
    # Smoothing
    # -------------------------
    def ema_update(self, name: str, fresh: BlockStats) -> BlockStats:
        if self.hist_enabled:
            return self._update_history(name, fresh)

        ours = (self.cfg.get("method", {}) or {}).get("ours", {}) or {}
        H = ours.get("ema_H", 1)
        try:
            H = int(H)
        except Exception:
            H = 1

        if H <= 1:
            return fresh

        beta = float(H - 1) / float(H)
        prev = self._ema.get(name, None)

        if prev is None:
            out = fresh
        else:
            out = BlockStats(
                C_r=float(beta * prev.C_r + (1.0 - beta) * fresh.C_r),
                C_R=float(beta * prev.C_R + (1.0 - beta) * fresh.C_R),
                A_b=float(beta * prev.A_b + (1.0 - beta) * fresh.A_b),
            )

        self._ema[name] = out
        return out

    def _weighted_average(self, seq: List[BlockStats]) -> BlockStats:
        n = len(seq)
        if n == 0:
            return BlockStats(0.0, 0.0, 1.0)

        if self.hist_weighting == "uniform":
            w = np.ones((n,), dtype=np.float64)
        elif self.hist_weighting == "linear":
            w = np.arange(1, n + 1, dtype=np.float64)
        else:
            beta = float(self.hist_beta)
            w = np.array([beta ** (n - 1 - i) for i in range(n)], dtype=np.float64)

        w = w / max(1e-12, float(np.sum(w)))

        Cr = float(np.sum([w[i] * float(seq[i].C_r) for i in range(n)]))
        CR = float(np.sum([w[i] * float(seq[i].C_R) for i in range(n)]))
        Ab = float(np.sum([w[i] * float(seq[i].A_b) for i in range(n)]))

        return BlockStats(C_r=Cr, C_R=CR, A_b=Ab)

    def _update_history(self, name: str, s: BlockStats) -> BlockStats:
        if not self.hist_enabled:
            return s

        dq = self._hist.get(name, None)
        if dq is None:
            dq = deque(maxlen=int(self.hist_window))
            self._hist[name] = dq

        dq.append(s)
        return self._weighted_average(list(dq))

    # -------------------------
    # Utils
    # -------------------------
    def _sigmoid(self, x: float) -> float:
        x = float(x)
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        else:
            z = np.exp(x)
            return float(z / (1.0 + z))

    def _cos(self, a: torch.Tensor, b: torch.Tensor) -> float:
        na = float(torch.norm(a).item())
        nb = float(torch.norm(b).item())
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(torch.dot(a, b).item() / (na * nb + self.eps))

    def _pad_or_trim(self, vec: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Safe shape align WITHOUT semantic repeat:
          - if shorter: pad zeros
          - if longer: trim
        """
        dim = int(dim)
        if vec.numel() == dim:
            return vec
        if vec.numel() > dim:
            return vec[:dim]
        pad = torch.zeros((dim - vec.numel(),), device=vec.device, dtype=vec.dtype)
        return torch.cat([vec, pad], dim=0)

    def _repeat_to_dim(self, head_vec: torch.Tensor, hi_dim: int) -> torch.Tensor:
        """
        Legacy debug fallback only (NOT mathematically meaningful).
        Keep it so old runs don't crash, but the main path should avoid it.
        """
        if head_vec.numel() == 0:
            return torch.zeros((hi_dim,), device=head_vec.device, dtype=head_vec.dtype)
        rep = int(np.ceil(hi_dim / head_vec.numel()))
        return head_vec.repeat(rep)[:hi_dim]

    def _split_branch_vec(
        self,
        mod: torch.nn.Module,
        branch: str,
        grad_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if branch == "r":
            A = mod.lora_A_r.detach()
            B = mod.lora_B_r.detach()
        elif branch == "hi":
            A = mod.lora_A_hi.detach()
            B = mod.lora_B_hi.detach()
        else:
            raise ValueError(f"unknown branch: {branch}")

        A_n = int(A.numel())
        B_n = int(B.numel())
        need = A_n + B_n
        if grad_vec.numel() != need:
            grad_vec = self._pad_or_trim(grad_vec, need)
        dA = grad_vec[:A_n].view_as(A)
        dB = grad_vec[A_n : A_n + B_n].view_as(B)
        return A, B, dA, dB

    def _branch_delta_w(
        self,
        mod: torch.nn.Module,
        branch: str,
        grad_vec: torch.Tensor,
    ) -> torch.Tensor:
        A, B, dA, dB = self._split_branch_vec(mod, branch, grad_vec)
        return (B @ dA) + (dB @ A)

    # -------------------------
    # Gates
    # -------------------------
    def noise_gates(self, s: BlockStats) -> Tuple[float, float]:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}
        ng = ours.get("gate0_noise", None)
        if not isinstance(ng, dict):
            raise RuntimeError("[ShakeAlign] missing method.ours.gate0_noise")

        tau_n = float(ng["tau"])
        kappa_n = float(ng["kappa"])

        b_r = self._sigmoid(kappa_n * (tau_n - float(s.C_r)))
        b_R = self._sigmoid(kappa_n * (tau_n - float(s.C_R)))

        return float(b_r), float(b_R)

    def trusted_reference(self, s: BlockStats) -> str:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}
        # Keep fallback aligned with configs/base.yaml default.
        delta = float(ours.get("routing_delta", 0.005))
        dC = float(s.C_r - s.C_R)
        return "r" if (dC >= delta) else "R"

    # -------------------------
    # P mapping (the missing compensation)
    # -------------------------
    def _compensation_P(
        self,
        mod: torch.nn.Module,
        res_head: torch.Tensor,
        *,
        lam: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute P(res_head) in ΔW semantics.

        res_head corresponds to head(g_R_exec) - g_r_exec which is a residual over
        the (A_r, B_r) gradient vector.

        Steps:
          1) split res_head -> dA_r, dB_r
          2) build dW_res ≈ B_r dA_r + dB_r A_r
          3) find (dA_hi, dB_hi) such that:
                B_hi dA_hi + dB_hi A_hi ≈ dW_res
             using ridge-stabilized closed forms.
        """
        device = res_head.device
        dtype = res_head.dtype

        Ar_n = int(mod.lora_A_r.numel())
        Br_n = int(mod.lora_B_r.numel())
        need = Ar_n + Br_n

        if res_head.numel() != need:
            res_head = self._pad_or_trim(res_head, need)

        dA_r = res_head[:Ar_n].view_as(mod.lora_A_r)
        dB_r = res_head[Ar_n : Ar_n + Br_n].view_as(mod.lora_B_r)

        # detach current factors (treat as constants for mapping)
        A_r = mod.lora_A_r.detach()
        B_r = mod.lora_B_r.detach()

        # ΔW residual (first-order)
        dW_res = (B_r @ dA_r) + (dB_r @ A_r)  # [out, in]

        A_hi = mod.lora_A_hi.detach()
        B_hi = mod.lora_B_hi.detach()
        hi = int(A_hi.shape[0])  # [hi, in]

        I_hi = torch.eye(hi, device=device, dtype=dtype)

        # Solve for dB_hi: dB_hi A_hi ≈ dW_res
        # dB_hi = dW_res A_hi^T (A_hi A_hi^T + lam I)^-1
        G_A = (A_hi @ A_hi.t()) + float(lam) * I_hi  # [hi,hi]
        RHS_B = (dW_res @ A_hi.t())  # [out,hi]
        dB_hi = torch.linalg.solve(G_A.t(), RHS_B.t()).t()  # [out,hi]

        # Solve for dA_hi: B_hi dA_hi ≈ dW_res
        # dA_hi = (B_hi^T B_hi + lam I)^-1 B_hi^T dW_res
        G_B = (B_hi.t() @ B_hi) + float(lam) * I_hi  # [hi,hi]
        RHS_A = (B_hi.t() @ dW_res)  # [hi,in]
        dA_hi = torch.linalg.solve(G_B, RHS_A)  # [hi,in]

        comp = torch.cat([dA_hi.reshape(-1), dB_hi.reshape(-1)], dim=0)
        return comp

    # -------------------------
    # Execution: in-place corrections
    # -------------------------
    @torch.no_grad()
    def apply_in_place_corrections(
        self,
        lora_modules: Dict[str, torch.nn.Module],
        stats: Dict[str, BlockStats],
        vote_sums: Dict[str, Dict[str, torch.Tensor]],
        debug: bool = False,
        grad_norm_trace: bool = False,
        debug_history: bool = False,
    ) -> Dict[str, Any]:
        ours = self.cfg.get("method", {}).get("ours", {}) or {}

        trig_cfg = ours.get("trigger_gate0", {}) or {}
        tau_N = float(trig_cfg.get("tau_N", 0.6))
        # Keep fallback aligned with configs/base.yaml default.
        tau_D = float(trig_cfg.get("tau_D", 0.0033914772))

        pull_cfg = ours.get("pulling", {}) or {}
        # Keep fallback aligned with configs/base.yaml default.
        gamma_pull = float(pull_cfg.get("gamma_pull", 0.30))
        k_pull = float(pull_cfg.get("k_pull", 8.0))

        # compensation knobs
        comp_cfg = ours.get("compensation", None)
        if not isinstance(comp_cfg, dict):
            raise RuntimeError("[ShakeAlign] missing method.ours.compensation")
        lam = float(comp_cfg["ridge_lambda"])
        enable_comp = bool(comp_cfg["enabled"])

        info: Dict[str, Any] = {
            "tau_N": tau_N,
            "tau_D": tau_D,
            "triggered_blocks": 0.0,
            "considered_blocks": 0.0,
            "gate0_trigger_rate": 0.0,
            "pull_to_r_blocks": 0.0,
            "pull_to_R_blocks": 0.0,
            "pull_to_r_rate": 0.0,
            "pull_to_R_rate": 0.0,
            "alpha_pull_mean": 0.0,
            "alpha_pull_to_r_mean": 0.0,
            "alpha_pull_to_R_mean": 0.0,
            "info_retention_ratio": 0.0,
            "residual_visibility": 0.0,
            "conflict_resolution_rate": 0.0,
            "expert_load_r": 0.0,
            "expert_load_R": 0.0,
            "load_cv": float("nan"),
            "load_max_min_ratio": float("nan"),
            "utilization_ratio": float("nan"),
            "routing_entropy": float("nan"),
            "routing_entropy_raw": float("nan"),
            "active_experts": 0.0,
            "expert_purity": float("nan"),
            "intra_expert_coherence": float("nan"),
            "intra_expert_conflict": float("nan"),
            "inter_expert_similarity": float("nan"),
        }
        if debug:
            info["per_block"] = {}
        if grad_norm_trace:
            info["per_block_grad_norm"] = {}
        if debug_history:
            info["per_block_history"] = {}

        triggered = 0
        considered = 0
        pull_to_r = 0
        pull_to_R = 0
        alpha_pull_sum = 0.0
        alpha_pull_to_r_sum = 0.0
        alpha_pull_to_R_sum = 0.0
        info_num_sum = 0.0
        info_den_sum = 0.0
        residual_hi_norm_sum = 0.0
        residual_total_norm_sum = 0.0
        conflict_delta_sum = 0.0
        conflict_count = 0
        route_counts = {"r": 0, "R": 0}
        route_exec: Dict[str, List[torch.Tensor]] = {"r": [], "R": []}
        route_purity_sum = 0.0

        for name, mod in lora_modules.items():
            if name not in stats or name not in vote_sums:
                continue
            if getattr(mod, "lora_A_r", None) is None:
                continue
            if (
                mod.lora_A_r.grad is None
                or mod.lora_B_r.grad is None
                or mod.lora_A_hi.grad is None
                or mod.lora_B_hi.grad is None
            ):
                continue

            s_raw = stats[name]
            s = self._update_history(name, s_raw) if self.hist_enabled else s_raw

            pack = vote_sums[name]
            vr = pack.get("votes_r", None)
            vhi = pack.get("votes_hi", None)
            if not isinstance(vr, torch.Tensor) or vr.numel() == 0:
                continue

            if not isinstance(vhi, torch.Tensor) or vhi.numel() == 0:
                # IMPORTANT: keep hi-dim consistent (zeros), not empty
                hi_dim = int(mod.lora_A_hi.numel() + mod.lora_B_hi.numel())
                vhi = torch.zeros((vr.shape[0], hi_dim), device=vr.device, dtype=vr.dtype)

            b_r, b_R = self.noise_gates(s)

            mean_r = vr.mean(dim=0)
            mean_hi = vhi.mean(dim=0)

            g_r_prime = mean_r
            g_R_prime = torch.cat([mean_r, mean_hi], dim=0)

            head_R = g_R_prime[: g_r_prime.numel()]

            D = 1.0 - self._cos(g_r_prime, head_R)
            N_summary = float(max(b_r, b_R))
            gate0 = (N_summary >= tau_N) or (D >= tau_D)

            chi_star = self.trusted_reference(s)
            considered += 1

            deltaC = float(s.C_r - s.C_R)
            insuff = self._sigmoid(float(k_pull) * float(deltaC))
            over = 1.0 - insuff
            alpha_pull = float(gamma_pull) * float(over) if gate0 else 0.0

            g_r_exec = g_r_prime.clone()
            g_R_exec = g_R_prime.clone()

            # pull
            if chi_star == "r":
                # R head is pulled toward r
                g_R_exec[: g_r_exec.numel()] = (1.0 - alpha_pull) * g_R_exec[: g_r_exec.numel()] + alpha_pull * g_r_exec
            else:
                # r is pulled toward R head
                g_r_exec = (1.0 - alpha_pull) * g_r_exec + alpha_pull * g_R_exec[: g_r_exec.numel()]

            # Split head/tail
            hi_dim = int(mod.lora_A_hi.numel() + mod.lora_B_hi.numel())
            head_dim = int(g_r_exec.numel())

            head_R_exec = g_R_exec[:head_dim]
            tail_R_exec = g_R_exec[head_dim:]

            # safe align tail dim WITHOUT repeat
            tail_R_exec = self._pad_or_trim(tail_R_exec, hi_dim)

            # ✅ compensation residual (paper step)
            res_head = head_R_exec - g_r_exec
            comp_hi = torch.zeros((hi_dim,), device=tail_R_exec.device, dtype=tail_R_exec.dtype)

            if enable_comp and gate0:
                try:
                    comp_hi = self._compensation_P(mod, res_head, lam=lam)
                    comp_hi = self._pad_or_trim(comp_hi, hi_dim)
                except Exception:
                    # fail-open: no compensation if solver fails
                    comp_hi = torch.zeros((hi_dim,), device=tail_R_exec.device, dtype=tail_R_exec.dtype)

            # -------------------------
            # Apply grads back (scale-invariant decisions, REAL updates)
            # -------------------------
            gr = g_r_exec

            Ar_n = int(mod.lora_A_r.grad.numel())
            Br_n = int(mod.lora_B_r.grad.numel())
            Dr = Ar_n + Br_n
            if gr.numel() != Dr:
                gr = self._pad_or_trim(gr, Dr)

            # ✅ scale back to REAL optimizer grads (r branch)
            # scaling_r = float(getattr(mod, "scaling_r", None) or getattr(mod, "scaling", 1.0))
            # gr = gr * scaling_r

            # mod.lora_A_r.grad.copy_(gr[:Ar_n].view_as(mod.lora_A_r.grad))
            # mod.lora_B_r.grad.copy_(gr[Ar_n : Ar_n + Br_n].view_as(mod.lora_B_r.grad))

            # ghi = tail_R_exec + comp_hi

            # Ahi_n = int(mod.lora_A_hi.grad.numel())
            # Bhi_n = int(mod.lora_B_hi.grad.numel())
            # Dhi = Ahi_n + Bhi_n
            # if ghi.numel() != Dhi:
            #     ghi = self._pad_or_trim(ghi, Dhi)

            # # ✅ scale back to REAL optimizer grads (hi branch)
            # scaling_hi = float(getattr(mod, "scaling_hi", None) or getattr(mod, "scaling", 1.0))

            # if not hasattr(mod, "scaling_r") or not hasattr(mod, "scaling_hi"):
            #     raise RuntimeError(f"[ShakeAlign] missing scaling_r/scaling_hi for {name}")

            # scaling_r = float(getattr(mod, "scaling_r"))
            # scaling_hi = float(getattr(mod, "scaling_hi"))

            # ghi = ghi * scaling_hi

            # mod.lora_A_hi.grad.copy_(ghi[:Ahi_n].view_as(mod.lora_A_hi.grad))
            # mod.lora_B_hi.grad.copy_(ghi[Ahi_n : Ahi_n + Bhi_n].view_as(mod.lora_B_hi.grad))


            # --- require scaling ---
            if not hasattr(mod, "scaling_r") or not hasattr(mod, "scaling_hi"):
                raise RuntimeError(f"[ShakeAlign] missing scaling_r/scaling_hi for {name}")

            scaling_r = float(getattr(mod, "scaling_r"))
            scaling_hi = float(getattr(mod, "scaling_hi"))
            if abs(scaling_r) < 1e-12 or abs(scaling_hi) < 1e-12:
                raise RuntimeError(f"[ScalingInvalid][ShakeAlign] scaling_r={scaling_r} scaling_hi={scaling_hi}")

            # --- write back r branch grads ---
            Ar_n = int(mod.lora_A_r.grad.numel())
            Br_n = int(mod.lora_B_r.grad.numel())
            Dr = Ar_n + Br_n
            gr = g_r_exec
            if gr.numel() != Dr:
                gr = self._pad_or_trim(gr, Dr)
            gr = gr * scaling_r
            mod.lora_A_r.grad.copy_(gr[:Ar_n].view_as(mod.lora_A_r.grad))
            mod.lora_B_r.grad.copy_(gr[Ar_n:Ar_n + Br_n].view_as(mod.lora_B_r.grad))

            # --- build hi branch grads ---
            ghi = tail_R_exec + comp_hi
            Ahi_n = int(mod.lora_A_hi.grad.numel())
            Bhi_n = int(mod.lora_B_hi.grad.numel())
            Dhi = Ahi_n + Bhi_n
            if ghi.numel() != Dhi:
                ghi = self._pad_or_trim(ghi, Dhi)

            dW_r_raw = self._branch_delta_w(mod, "r", g_r_prime)
            dW_hi_raw = self._branch_delta_w(mod, "hi", mean_hi)
            dW_r_exec = self._branch_delta_w(mod, "r", g_r_exec)
            dW_hi_exec = self._branch_delta_w(mod, "hi", ghi)

            dW_raw = dW_r_raw + dW_hi_raw
            dW_exec = dW_r_exec + dW_hi_exec
            raw_flat = dW_raw.reshape(-1)
            exec_flat = dW_exec.reshape(-1)
            raw_norm_sq = float(torch.dot(raw_flat, raw_flat).item())
            if raw_norm_sq > self.eps:
                info_num_sum += float(torch.dot(exec_flat, raw_flat).item())
                info_den_sum += raw_norm_sq

            hi_exec_norm = float(torch.norm(dW_hi_exec).item())
            r_exec_norm = float(torch.norm(dW_r_exec).item())
            residual_hi_norm_sum += hi_exec_norm
            residual_total_norm_sum += (r_exec_norm + hi_exec_norm)

            conflict_before = self._cos(dW_r_raw.reshape(-1), dW_hi_raw.reshape(-1))
            conflict_after = self._cos(dW_r_exec.reshape(-1), dW_hi_exec.reshape(-1))
            conflict_delta_sum += float(conflict_after - conflict_before)
            conflict_count += 1

            ghi = ghi * scaling_hi
            mod.lora_A_hi.grad.copy_(ghi[:Ahi_n].view_as(mod.lora_A_hi.grad))
            mod.lora_B_hi.grad.copy_(ghi[Ahi_n:Ahi_n + Bhi_n].view_as(mod.lora_B_hi.grad))


            if gate0:
                triggered += 1
                alpha_pull_sum += float(alpha_pull)
                route_counts[chi_star] += 1
                route_exec[chi_star].append(exec_flat.detach().clone())
                route_purity_sum += float(max(insuff, over))
                if chi_star == "r":
                    pull_to_r += 1
                    alpha_pull_to_r_sum += float(alpha_pull)
                else:
                    pull_to_R += 1
                    alpha_pull_to_R_sum += float(alpha_pull)

            if debug:
                info["per_block"][name] = {
                    "C_r": float(s.C_r),
                    "C_R": float(s.C_R),
                    "A_b": float(s.A_b),
                    "deltaC": float(deltaC),
                    "b_r": float(b_r),
                    "b_R": float(b_R),
                    "N_summary": float(N_summary),
                    "D": float(D),
                    "gate0": bool(gate0),
                    "chi_star": str(chi_star),
                    "alpha_pull": float(alpha_pull),
                    "comp_enabled": bool(enable_comp),
                    "ridge_lambda": float(lam),
                }

            if grad_norm_trace:
                info["per_block_grad_norm"][name] = {
                    "||g_r'||": float(torch.norm(g_r_prime).item()),
                    "||head(g_R')||": float(torch.norm(head_R).item()),
                    "||tail(g_R_exec)||": float(torch.norm(tail_R_exec).item()),
                    "||P(res_head)||": float(torch.norm(comp_hi).item()),
                    "scaling_r": float(scaling_r),
                    "scaling_hi": float(scaling_hi),
                }

            if debug_history and self.hist_enabled:
                dq = self._hist.get(name, None)
                if dq is not None:
                    info["per_block_history"][name] = [
                        {"C_r": float(x.C_r), "C_R": float(x.C_R), "A_b": float(x.A_b)} for x in list(dq)
                    ]

        info["triggered_blocks"] = float(triggered)
        info["considered_blocks"] = float(considered)
        if considered > 0:
            info["gate0_trigger_rate"] = float(triggered) / float(considered)
        if triggered > 0:
            info["pull_to_r_blocks"] = float(pull_to_r)
            info["pull_to_R_blocks"] = float(pull_to_R)
            info["pull_to_r_rate"] = float(pull_to_r) / float(triggered)
            info["pull_to_R_rate"] = float(pull_to_R) / float(triggered)
            info["alpha_pull_mean"] = float(alpha_pull_sum) / float(triggered)
            if pull_to_r > 0:
                info["alpha_pull_to_r_mean"] = float(alpha_pull_to_r_sum) / float(pull_to_r)
            if pull_to_R > 0:
                info["alpha_pull_to_R_mean"] = float(alpha_pull_to_R_sum) / float(pull_to_R)
        if info_den_sum > self.eps:
            info["info_retention_ratio"] = float(info_num_sum / info_den_sum)
        if residual_total_norm_sum > self.eps:
            info["residual_visibility"] = float(residual_hi_norm_sum / residual_total_norm_sum)
        if conflict_count > 0:
            info["conflict_resolution_rate"] = float(conflict_delta_sum / float(conflict_count))
        if triggered > 0:
            load_r = int(route_counts["r"])
            load_R = int(route_counts["R"])
            loads = np.asarray([load_r, load_R], dtype=np.float64)
            info["expert_load_r"] = float(load_r)
            info["expert_load_R"] = float(load_R)
            info["active_experts"] = float(np.count_nonzero(loads > 0.0))
            mean_load = float(np.mean(loads))
            if mean_load > 0.0:
                info["load_cv"] = float(np.std(loads) / mean_load)
            min_load = float(np.min(loads))
            max_load = float(np.max(loads))
            if min_load > 0.0:
                info["load_max_min_ratio"] = float(max_load / min_load)
            info["utilization_ratio"] = float(np.count_nonzero(loads > 0.0) / float(loads.size))
            probs = loads / max(1.0, float(loads.sum()))
            raw_entropy = float(-np.sum([p * np.log(p + self.eps) for p in probs if p > 0.0]))
            info["routing_entropy_raw"] = raw_entropy
            if loads.size > 1:
                info["routing_entropy"] = float(raw_entropy / np.log(float(loads.size)))
            info["expert_purity"] = float(route_purity_sum / float(triggered))

            intra_vals: List[float] = []
            for dest in ("r", "R"):
                v = _mean_pairwise_cos(route_exec[dest], self.eps)
                if np.isfinite(v):
                    intra_vals.append(float(v))
            if intra_vals:
                intra_coh = float(np.mean(intra_vals))
                info["intra_expert_coherence"] = intra_coh
                info["intra_expert_conflict"] = float(0.5 * (1.0 - intra_coh))

            inter_sim = _mean_cross_bucket_cos(route_exec["r"], route_exec["R"], self.eps)
            if np.isfinite(inter_sim):
                info["inter_expert_similarity"] = float(inter_sim)
        return info
