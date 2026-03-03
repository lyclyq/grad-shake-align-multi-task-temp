#!/usr/bin/env python3
# /home/lyclyq/Optimization/grad-shake-align/scripts/pipeline_oneclick.py
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------
# Helpers: timestamp/hash binding
# -------------------------
_TS_RE = re.compile(r"__(\d{8}-\d{6})__")          # matches __YYYYMMDD-HHMMSS__
_HASH_RE = re.compile(r"__([0-9a-f]{6,12})$")     # matches __8d329d88 at end


def _extract_ts_and_hash_from_dirname(name: str) -> Tuple[str, str]:
    """
    Extract timestamp + short hash from a run directory name.
    Returns ("20260205-120134", "8d329d88") or ("unknownTS","unknownHASH")
    """
    m_ts = _TS_RE.search(name)
    m_h = _HASH_RE.search(name)
    ts = m_ts.group(1) if m_ts else "unknownTS"
    hh = m_h.group(1) if m_h else "unknownHASH"
    return ts, hh


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(s)).strip("-").lower()


def _default_runs_group(dataset: str, model: str) -> str:
    ds = dataset.replace("/", "_")
    if "roberta" in model.lower():
        bb = "roberta"
    elif "bert" in model.lower():
        bb = "bert"
    else:
        bb = model.split("-")[0]
    return f"{ds}_{bb}"


@dataclass
class ExpSpec:
    runs_group: str
    final_epochs: Optional[int]
    final_seeds: List[int]
    resume_debug: str

    # optional overrides (passed through as --set)
    dataset: Optional[str] = None
    model: Optional[str] = None
    trials: Optional[int] = None

    # ranks (must sync baseline_r/baseline_R/ours)
    ours_r: Optional[int] = None
    ours_R: Optional[int] = None

    ablate_interp: bool = False
    history_enabled: bool = False
    rerank_enabled: Optional[bool] = None
    rerank_top_k: Optional[int] = None
    rerank_epochs: Optional[int] = None


def sh(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def find_latest_hpo_dir(parent: Path) -> Path:
    # Support both legacy debug__* and current hpo__* naming.
    cands = [
        p for p in parent.iterdir()
        if p.is_dir() and (p.name.startswith("hpo__") or p.name.startswith("debug__"))
    ]
    if not cands:
        raise FileNotFoundError(f"No HPO dir (hpo__/debug__) under {parent}")

    # Prefer dirs that already have best_hparams.json, then newest by mtime.
    with_best = [p for p in cands if (p / "best_hparams.json").exists()]
    pool = with_best if with_best else cands
    return max(pool, key=lambda p: p.stat().st_mtime)


def dump_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _parse_seeds(s: Optional[str]) -> List[int]:
    if s is None:
        return []
    ss = str(s).strip()
    if not ss:
        return []
    if ss.startswith("["):
        return [int(x) for x in json.loads(ss)]
    return [int(x) for x in ss.split(",") if x.strip()]


def _parse_bool_arg(s: str) -> bool:
    x = str(s).strip().lower()
    if x in {"1", "true", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {s!r} (use true/false)")


def _get_nested(obj: Any, dotted: str) -> Any:
    cur = obj
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _check_resume_snapshot_compat(
    *,
    debug_dir: Path,
    base_cfg: str,
    hpo_set_args: List[str],
) -> None:
    snap = debug_dir / "config_snapshot.json"
    if not snap.exists():
        return

    from src.config import load_config_with_cli_overrides

    cfg_now = load_config_with_cli_overrides(base_cfg, None, hpo_set_args)
    cfg_old = json.loads(snap.read_text(encoding="utf-8"))

    keys = [
        "task.name",
        "model.name",
        "hpo.budget.total_trials",
        "hpo.grid.rerank.enabled",
        "hpo.grid.rerank.top_k",
        "hpo.grid.rerank.epochs",
        "method.baseline_r.lora.r",
        "method.baseline_R.lora.r",
        "method.ours.lora.r",
        "method.ours.lora.R",
    ]

    mismatches: List[str] = []
    for k in keys:
        old_v = _get_nested(cfg_old, k)
        now_v = _get_nested(cfg_now, k)
        if old_v != now_v:
            mismatches.append(f"{k}: old={old_v!r}, now={now_v!r}")

    if mismatches:
        detail = "\n".join(mismatches)
        raise RuntimeError(
            "[PIPE] resume_debug config mismatch detected. "
            "Refuse to resume mixed HPO states.\n"
            f"[PIPE] debug_dir={debug_dir}\n"
            "[PIPE] mismatches:\n"
            f"{detail}\n"
            "[PIPE] Use a fresh HPO run (no --resume_debug), or resume with identical HPO/rank/task/model settings."
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline_oneclick.py",
        description="One-click pipeline with resume support (NO YAML parsing here; config.py is the only config loader).",
    )

    # Optional pass-through overrides (ONLY applied if provided)
    p.add_argument("--dataset", type=str, default=None, help="Override task.name via --set task.name=...")
    p.add_argument("--model", type=str, default=None, help="Override model.name via --set model.name=...")
    p.add_argument("--trials", type=int, default=None, help="Override hpo.budget.total_trials via --set")
    p.add_argument(
        "--rerank_enabled",
        type=_parse_bool_arg,
        default=None,
        help="Override hpo.grid.rerank.enabled via --set (true/false)",
    )
    p.add_argument("--rerank_top_k", type=int, default=None, help="Override hpo.grid.rerank.top_k via --set")
    p.add_argument("--rerank_epochs", type=int, default=None, help="Override hpo.grid.rerank.epochs via --set")

    # ranks: STRICT sync for baseline_r/baseline_R/ours
    p.add_argument("--ours_r", type=int, default=None, help="Override small rank r (sync baseline_r + ours.r)")
    p.add_argument("--ours_R", type=int, default=None, help="Override large rank R (sync baseline_R + ours.R)")

    # Pipeline-only (output organization)
    p.add_argument(
        "--runs_group",
        type=str,
        default=None,
        help="Runs group folder under runs/. If omitted, derive from effective task/model (CLI overrides > base config).",
    )

    # Optional overrides for final stage ONLY
    p.add_argument("--final_epochs", type=int, default=None, help="Override train.epochs for final only via --set")
    p.add_argument("--final_seeds", type=str, default=None, help="Override final.seeds for final only via --set")

    p.add_argument("--ablate_interp", action="store_true")
    p.add_argument("--history_enabled", action="store_true")

    p.add_argument(
        "--resume_debug",
        type=str,
        default="",
        help="Resume an existing HPO directory (hpo__*/debug__*, absolute path). If empty, run HPO normally.",
    )

    return p


def main() -> int:
    args = build_argparser().parse_args()
    base_cfg = "configs/base.yaml"
    final_schedule = "configs/schedules/final.yaml"

    # -------------------------
    # Resolve runs_group
    # -------------------------
    runs_group: Optional[str] = None
    if isinstance(args.runs_group, str) and args.runs_group.strip():
        runs_group = args.runs_group.strip()
    else:
        derive_sets: List[str] = []
        if args.dataset is not None:
            derive_sets.append(f"task.name={str(args.dataset)}")
        if args.model is not None:
            derive_sets.append(f"model.name={str(args.model)}")
        try:
            from src.config import load_config_with_cli_overrides

            cfg_eff = load_config_with_cli_overrides(base_cfg, None, derive_sets)
            ds_eff = str(cfg_eff["task"]["name"])
            md_eff = str(cfg_eff["model"]["name"])
            runs_group = _default_runs_group(ds_eff, md_eff)
            print(f"[PIPE] runs_group auto-derived from config: {runs_group}")
        except Exception as e:
            raise RuntimeError(
                "[PIPE] failed to derive runs_group from base config "
                f"(root cause: {type(e).__name__}: {e}). "
                "Use the correct runtime env (e.g., conda optimization), "
                "or set --runs_group explicitly."
            ) from e

    spec = ExpSpec(
        runs_group=runs_group,
        final_epochs=int(args.final_epochs) if args.final_epochs is not None else None,
        final_seeds=_parse_seeds(args.final_seeds),
        resume_debug=str(args.resume_debug or ""),
        dataset=str(args.dataset) if args.dataset is not None else None,
        model=str(args.model) if args.model is not None else None,
        trials=int(args.trials) if args.trials is not None else None,
        rerank_enabled=args.rerank_enabled,
        rerank_top_k=int(args.rerank_top_k) if args.rerank_top_k is not None else None,
        rerank_epochs=int(args.rerank_epochs) if args.rerank_epochs is not None else None,
        ours_r=int(args.ours_r) if args.ours_r is not None else None,
        ours_R=int(args.ours_R) if args.ours_R is not None else None,
        ablate_interp=bool(args.ablate_interp),
        history_enabled=bool(args.history_enabled),
    )

    if spec.rerank_top_k is not None and spec.rerank_top_k <= 0:
        raise ValueError("--rerank_top_k must be > 0")
    if spec.rerank_epochs is not None and spec.rerank_epochs <= 0:
        raise ValueError("--rerank_epochs must be > 0")

    runs_root = ROOT / "runs" / spec.runs_group

    # -------------------------
    # Sets split: GLOBAL vs OURS-ONLY vs HPO-ONLY
    # -------------------------
    # GLOBAL: safe for both ours and baselines
    common_sets_global: List[str] = [
        f"io.root={str(runs_root)}",
    ]
    if spec.dataset is not None:
        common_sets_global.append(f"task.name={spec.dataset}")
    if spec.model is not None:
        common_sets_global.append(f"model.name={spec.model}")

    # ✅ STRICT rank sync (no method.lora.* anywhere)
    if spec.ours_r is not None:
        # baseline_r uses r field
        common_sets_global += [
            f"method.baseline_r.lora.r={spec.ours_r}",
            f"method.ours.lora.r={spec.ours_r}",
        ]
    if spec.ours_R is not None:
        # baseline_R uses r field to store its rank (large rank)
        common_sets_global += [
            f"method.baseline_R.lora.r={spec.ours_R}",
            f"method.ours.lora.R={spec.ours_R}",
        ]

    # OURS-ONLY: only for HPO stage (FINAL must not receive these)
    common_sets_ours_only: List[str] = []
    if spec.ablate_interp:
        common_sets_ours_only.append("method.ours.ablate.interp=true")
    if spec.history_enabled:
        common_sets_ours_only.append("method.ours.history.enabled=true")

    # HPO-ONLY
    hpo_sets_only: List[str] = []
    if spec.trials is not None:
        hpo_sets_only.append(f"hpo.budget.total_trials={spec.trials}")
    if spec.rerank_enabled is not None:
        hpo_sets_only.append(f"hpo.grid.rerank.enabled={str(spec.rerank_enabled).lower()}")
    if spec.rerank_top_k is not None:
        hpo_sets_only.append(f"hpo.grid.rerank.top_k={spec.rerank_top_k}")
    if spec.rerank_epochs is not None:
        hpo_sets_only.append(f"hpo.grid.rerank.epochs={spec.rerank_epochs}")

    # ---------------- (1) HPO ----------------
    cmd_hpo = ["python", "scripts/run.py", "hpo", "--config", base_cfg]
    for s in (common_sets_global + common_sets_ours_only + hpo_sets_only):
        cmd_hpo += ["--set", s]

    if spec.resume_debug:
        debug_dir = Path(spec.resume_debug).resolve()
        _check_resume_snapshot_compat(
            debug_dir=debug_dir,
            base_cfg=base_cfg,
            hpo_set_args=(common_sets_global + common_sets_ours_only + hpo_sets_only),
        )
        cmd_hpo += ["--set", f"io.run_dir={str(debug_dir)}"]
        cmd_hpo += ["--set", "io.overwrite=resume"]
        print(f"[PIPE] Resuming HPO from {debug_dir}")
    else:
        debug_dir = None

    sh(cmd_hpo)

    if debug_dir is None:
        debug_dir = find_latest_hpo_dir(runs_root)

    print("[INFO] debug_dir =", debug_dir)

    best_path = debug_dir / "best_hparams.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing: {best_path}")

    # ---------------- (2) Final (timestamp-bound, deterministic) ----------------
    dbg_ts, dbg_hash = _extract_ts_and_hash_from_dirname(debug_dir.name)
    final_dir = runs_root / (f"final__{_slug(spec.runs_group)}__from_{dbg_ts}__{dbg_hash}")

    cmd_final = [
        "python", "scripts/run.py", "final",
        "--config", base_cfg,
        "--schedule", final_schedule,
        "--best", str(best_path),
    ]

    # FINAL gets ONLY GLOBAL sets (never ours-only, never hpo-only)
    for s in common_sets_global:
        cmd_final += ["--set", s]

    # Optional overrides: only if user explicitly provided
    if spec.final_epochs is not None:
        cmd_final += ["--set", f"train.epochs={spec.final_epochs}"]
        cmd_final += ["--set", f"final.epochs={spec.final_epochs}"]
    if spec.final_seeds:
        cmd_final += ["--set", "final.seeds=" + json.dumps(spec.final_seeds)]

    # deterministic final_dir + resume semantics
    cmd_final += ["--set", f"io.run_dir={str(final_dir)}"]
    cmd_final += ["--set", "io.overwrite=resume"]

    sh(cmd_final)

    print("[INFO] final_dir =", final_dir)

    # pipeline-level provenance marker
    try:
        dump_json(
            final_dir / "final_provenance_from_pipeline.json",
            {
                "debug_dir": str(debug_dir),
                "debug_name": debug_dir.name,
                "debug_timestamp": dbg_ts,
                "debug_hash": dbg_hash,
                "best_path": str(best_path),
                "final_dir": str(final_dir),
                "spec": {
                    "runs_group": spec.runs_group,
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "total_trials": spec.trials,
                    "rerank_enabled": spec.rerank_enabled,
                    "rerank_top_k": spec.rerank_top_k,
                    "rerank_epochs": spec.rerank_epochs,
                    "ours_r": spec.ours_r,
                    "ours_R": spec.ours_R,
                    "final_epochs": spec.final_epochs,
                    "final_seeds": spec.final_seeds,
                    "ablate_interp": spec.ablate_interp,
                    "history_enabled": spec.history_enabled,
                },
                "sets": {
                    "global": common_sets_global,
                    "ours_only": common_sets_ours_only,
                    "hpo_only": hpo_sets_only,
                },
            },
        )
    except Exception as e:
        print(f"[PIPE][WARN] failed to write provenance json: {e}")

    # ---------------- (3) Plot ----------------
    trial_runs = final_dir / "trial_runs"
    if not trial_runs.exists():
        raise FileNotFoundError(f"trial_runs_dir not found: {trial_runs}")

    # generic per-metric compare plots from all_curves -> plots_compare/
    plot_compare_cmd = [
        "python", "scripts/run.py", "plot",
        "--config", base_cfg,
        "--runs_dir", str(final_dir),
    ]
    sh(plot_compare_cmd)

    # final paper-style 4-line + ours diagnostics
    plot_4lines_cmd = ["python", "scripts/plot_final_4lines_abs.py", str(trial_runs), "val/acc"]
    sh(plot_4lines_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
