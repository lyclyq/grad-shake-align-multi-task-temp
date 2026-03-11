#!/usr/bin/env python3
# /home/lyclyq/Optimization/grad-shake-align/scripts/pipeline_oneclick.py
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
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


def _default_runs_group_from_cfg(cfg: dict) -> str:
    task = cfg["task"]
    model = str(cfg["model"]["name"])
    scenario = str(task.get("scenario", "single_task")).strip().lower()
    multi = (task.get("multi", {}) or {}) if isinstance(task, dict) else {}
    if bool(multi.get("enabled", False)):
        ds_list = list(multi.get("datasets", []) or [])
        ds = "+".join([str(x).replace("/", "_") for x in ds_list]) if ds_list else str(task["name"]).replace("/", "_")
    else:
        ds = str(task["name"]).replace("/", "_")

    if "roberta" in model.lower():
        bb = "roberta"
    elif "bert" in model.lower():
        bb = "bert"
    else:
        bb = model.split("-")[0]
    return f"{scenario}__{ds}_{bb}"


def _multi_ds_token(ds_list: List[str]) -> str:
    xs = [str(x).strip().replace("/", "_") for x in (ds_list or []) if str(x).strip()]
    return "+".join(xs)


def _augment_runs_group_with_datasets(runs_group: str, *, multi_enabled: Optional[bool], multi_datasets: Optional[List[str]]) -> str:
    if not bool(multi_enabled):
        return str(runs_group)
    ds_tok = _multi_ds_token(list(multi_datasets or []))
    if not ds_tok:
        return str(runs_group)
    rg = str(runs_group)
    if ds_tok in rg:
        return rg
    return f"{rg}__{ds_tok}"


def _fmt_stage_budget_tag(max_steps: Optional[int], epochs: Optional[int], prefix: str) -> str:
    if max_steps is not None:
        return f"{prefix}ms{int(max_steps)}"
    if epochs is not None:
        return f"{prefix}ep{int(epochs)}"
    return f"{prefix}base"


def _pipeline_budget_tag(spec: "ExpSpec") -> str:
    ttag = f"t{int(spec.trials)}" if spec.trials is not None else "tbase"
    run_tag = _fmt_stage_budget_tag(
        spec.max_steps if spec.multi_steps_mode == "max_steps" else None,
        spec.epochs if spec.multi_steps_mode == "epochs" else None,
        "run",
    )
    hbs = _fmt_stage_budget_tag(spec.hpo_baseline_max_steps, spec.hpo_baseline_epochs, "hbs")
    hg = _fmt_stage_budget_tag(spec.hpo_grid_max_steps, spec.hpo_grid_epochs, "hg")
    hrr = _fmt_stage_budget_tag(spec.hpo_rerank_max_steps, spec.rerank_epochs, "hrr")
    ftag = f"fep{int(spec.final_epochs)}" if spec.final_epochs is not None else "fepbase"
    return f"{ttag}_{run_tag}_{hbs}_{hg}_{hrr}_{ftag}"


@dataclass
class ExpSpec:
    runs_group: str
    epochs: Optional[int]
    final_epochs: Optional[int]
    final_seeds: List[int]
    resume_debug: str

    # optional overrides (passed through as --set)
    scenario: Optional[str] = None
    dataset: Optional[str] = None
    model: Optional[str] = None
    trials: Optional[int] = None
    multi_enabled: Optional[bool] = None
    multi_datasets: Optional[List[str]] = None
    multi_steps_mode: Optional[str] = None
    max_steps: Optional[int] = None
    multi_drop_last: Optional[bool] = None

    # ranks (must sync baseline_r/baseline_R/ours)
    ours_r: Optional[int] = None
    ours_R: Optional[int] = None

    ablate_interp: bool = False
    history_enabled: bool = False
    rerank_enabled: Optional[bool] = None
    rerank_top_k: Optional[int] = None
    rerank_epochs: Optional[int] = None
    hpo_baseline_epochs: Optional[int] = None
    hpo_grid_epochs: Optional[int] = None
    hpo_baseline_max_steps: Optional[int] = None
    hpo_grid_max_steps: Optional[int] = None
    hpo_rerank_max_steps: Optional[int] = None


def sh(cmd: List[str], cwd: Optional[Path] = None) -> None:
    if cmd and cmd[0] == "python":
        cmd = [sys.executable] + cmd[1:]
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


def _parse_datasets_arg(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    ss = str(s).strip()
    if not ss:
        return []
    if ss.startswith("["):
        vals = json.loads(ss)
        if not isinstance(vals, list):
            raise argparse.ArgumentTypeError("--multi_datasets JSON must be a list")
        out = [str(x).strip() for x in vals if str(x).strip()]
    else:
        out = [x.strip() for x in ss.split(",") if x.strip()]
    return out


def _parse_mixtures_arg(s: Optional[str]) -> List[List[str]]:
    if s is None:
        return []
    ss = str(s).strip()
    if not ss:
        return []
    vals = json.loads(ss)
    if not isinstance(vals, list):
        raise argparse.ArgumentTypeError("--multi_mixtures must be a JSON list")
    out: List[List[str]] = []
    for i, item in enumerate(vals):
        if not isinstance(item, list):
            raise argparse.ArgumentTypeError(f"--multi_mixtures[{i}] must be a list")
        row = [str(x).strip() for x in item if str(x).strip()]
        if not row:
            raise argparse.ArgumentTypeError(f"--multi_mixtures[{i}] must be non-empty")
        out.append(row)
    return out


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
        "task.multi.enabled",
        "task.multi.datasets",
        "train.single.steps_mode",
        "train.multi.steps_mode",
        "model.name",
        "train.max_steps",
        "train.epochs",
        "train.multi.drop_last",
        "hpo.budget.total_trials",
        "hpo.grid.rerank.enabled",
        "hpo.grid.rerank.top_k",
        "hpo.grid.rerank.epochs",
        "hpo.grid.baseline_search_epochs",
        "hpo.grid.grid_epochs",
        "hpo.grid.baseline_search_max_steps",
        "hpo.grid.grid_max_steps",
        "hpo.grid.rerank.max_steps",
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


def _check_resume_final_compat(*, final_dir: Path, expected_spec: "ExpSpec", with_ablations: bool) -> None:
    prov = final_dir / "final_provenance_from_pipeline.json"
    if not prov.exists():
        raise RuntimeError(f"[PIPE] missing final provenance for resume: {prov}")
    obj = json.loads(prov.read_text(encoding="utf-8"))
    spec_old = obj.get("spec", {}) if isinstance(obj, dict) else {}
    checks = {
        "runs_group": expected_spec.runs_group,
        "scenario": expected_spec.scenario,
        "dataset": expected_spec.dataset,
        "model": expected_spec.model,
        "multi_enabled": expected_spec.multi_enabled,
        "multi_datasets": expected_spec.multi_datasets,
        "multi_steps_mode": expected_spec.multi_steps_mode,
        "max_steps": expected_spec.max_steps,
        "epochs": expected_spec.epochs,
        "ours_r": expected_spec.ours_r,
        "ours_R": expected_spec.ours_R,
        "with_ablations": bool(with_ablations),
    }
    mismatches: List[str] = []
    for k, now_v in checks.items():
        old_v = spec_old.get(k, None)
        if now_v is not None and old_v != now_v:
            mismatches.append(f"{k}: old={old_v!r}, now={now_v!r}")
    if mismatches:
        raise RuntimeError(
            "[PIPE] resume_final config mismatch detected.\n"
            f"[PIPE] final_dir={final_dir}\n"
            "[PIPE] mismatches:\n"
            + "\n".join(mismatches)
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline_oneclick.py",
        description="One-click pipeline with resume support (NO YAML parsing here; config.py is the only config loader).",
    )

    # Optional pass-through overrides (ONLY applied if provided)
    p.add_argument("--dataset", type=str, default=None, help="Override task.name via --set task.name=...")
    p.add_argument("--model", type=str, default=None, help="Override model.name via --set model.name=...")
    p.add_argument(
        "--scenario",
        type=str,
        choices=["single_task", "multi_task", "multi_dataset"],
        default=None,
        help="Override task.scenario via --set",
    )
    p.add_argument(
        "--multi_enabled",
        type=_parse_bool_arg,
        default=None,
        help="Override task.multi.enabled via --set (true/false)",
    )
    p.add_argument(
        "--multi_datasets",
        type=str,
        default=None,
        help="Override task.multi.datasets via --set. Accept JSON list or comma list.",
    )
    p.add_argument(
        "--multi_mixtures",
        type=str,
        default=None,
        help="Batch mode: JSON list of dataset lists. Each mixture launches one pipeline run.",
    )
    p.add_argument(
        "--multi_steps_mode",
        type=str,
        choices=["max_steps", "epochs"],
        default=None,
        help="Override train.multi.steps_mode via --set (max_steps/epochs)",
    )
    p.add_argument("--max_steps", type=int, default=None, help="Override train.max_steps via --set (for multi-task)")
    p.add_argument(
        "--multi_drop_last",
        type=_parse_bool_arg,
        default=None,
        help="Override train.multi.drop_last via --set (true/false)",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override hpo.budget.total_trials via --set. Recommended tiers: 48 / 96 / 192. Default: 96.",
    )
    p.add_argument(
        "--rerank_enabled",
        type=_parse_bool_arg,
        default=None,
        help="Override hpo.grid.rerank.enabled via --set (true/false)",
    )
    p.add_argument("--rerank_top_k", type=int, default=None, help="Override hpo.grid.rerank.top_k via --set")
    p.add_argument("--rerank_epochs", type=int, default=None, help="Override hpo.grid.rerank.epochs via --set")
    p.add_argument("--hpo_baseline_epochs", type=int, default=None, help="Override hpo.grid.baseline_search_epochs via --set")
    p.add_argument("--hpo_grid_epochs", type=int, default=None, help="Override hpo.grid.grid_epochs via --set")
    p.add_argument(
        "--hpo_baseline_max_steps",
        type=int,
        default=None,
        help="Override hpo.grid.baseline_search_max_steps via --set",
    )
    p.add_argument("--hpo_grid_max_steps", type=int, default=None, help="Override hpo.grid.grid_max_steps via --set")
    p.add_argument("--hpo_rerank_max_steps", type=int, default=None, help="Override hpo.grid.rerank.max_steps via --set")

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
    p.add_argument("--epochs", type=int, default=None, help="Override train.epochs globally via --set")

    # Optional overrides for final stage ONLY
    p.add_argument("--final_epochs", type=int, default=None, help="Override train.epochs for final only via --set")
    p.add_argument("--final_seeds", type=str, default=None, help="Override final.seeds for final only via --set")

    p.add_argument("--ablate_interp", action="store_true")
    p.add_argument("--history_enabled", action="store_true")
    p.add_argument("--with_ablations", action="store_true")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Extra raw --set overrides passthrough to HPO+FINAL (repeatable, e.g. --set train.batch_size=64).",
    )

    p.add_argument(
        "--resume_debug",
        type=str,
        default="",
        help="Resume an existing HPO directory (hpo__*/debug__*, absolute path). If empty, run HPO normally.",
    )
    p.add_argument(
        "--resume_final",
        type=str,
        default="",
        help="Resume an existing final directory. If set, pipeline reuses that final_dir and only fills missing runs.",
    )

    return p


def main() -> int:
    args = build_argparser().parse_args()
    mixtures = _parse_mixtures_arg(args.multi_mixtures)
    if mixtures:
        raw_argv = list(sys.argv[1:])
        strip_next_for = {"--multi_mixtures", "--multi_datasets", "--multi_enabled"}
        filtered: List[str] = []
        skip = False
        for tok in raw_argv:
            if skip:
                skip = False
                continue
            if tok in strip_next_for:
                skip = True
                continue
            filtered.append(tok)
        for mix in mixtures:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                *filtered,
                "--scenario", "multi_task",
                "--multi_enabled", "true",
                "--multi_datasets", json.dumps(mix),
            ]
            print(f"[PIPE][BATCH] mixture={mix}")
            subprocess.run(cmd, cwd=str(ROOT), check=True)
        return 0

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
        if args.scenario is not None:
            derive_sets.append(f"task.scenario={str(args.scenario)}")
        if args.multi_enabled is not None:
            derive_sets.append(f"task.multi.enabled={str(args.multi_enabled).lower()}")
        ds_cli = _parse_datasets_arg(args.multi_datasets)
        if ds_cli:
            derive_sets.append("task.multi.datasets=" + json.dumps(ds_cli))
        if args.max_steps is not None:
            derive_sets.append(f"train.max_steps={int(args.max_steps)}")
        if args.multi_steps_mode is not None:
            derive_sets.append(f"train.multi.steps_mode={str(args.multi_steps_mode)}")
        if args.multi_drop_last is not None:
            derive_sets.append(f"train.multi.drop_last={str(args.multi_drop_last).lower()}")
        if args.epochs is not None:
            derive_sets.append(f"train.epochs={int(args.epochs)}")
        try:
            from src.config import load_config_with_cli_overrides

            cfg_eff = load_config_with_cli_overrides(base_cfg, None, derive_sets)
            runs_group = _default_runs_group_from_cfg(cfg_eff)
            print(f"[PIPE] runs_group auto-derived from config: {runs_group}")
        except Exception as e:
            raise RuntimeError(
                "[PIPE] failed to derive runs_group from base config "
                f"(root cause: {type(e).__name__}: {e}). "
                "Use the correct runtime env (e.g., conda optimization), "
                "or set --runs_group explicitly."
            ) from e

    ds_cli = _parse_datasets_arg(args.multi_datasets)
    spec = ExpSpec(
        runs_group=runs_group,
        epochs=int(args.epochs) if args.epochs is not None else None,
        final_epochs=int(args.final_epochs) if args.final_epochs is not None else None,
        final_seeds=_parse_seeds(args.final_seeds),
        resume_debug=str(args.resume_debug or ""),
        scenario=str(args.scenario) if args.scenario is not None else None,
        dataset=str(args.dataset) if args.dataset is not None else None,
        model=str(args.model) if args.model is not None else None,
        trials=int(args.trials) if args.trials is not None else None,
        multi_enabled=args.multi_enabled,
        multi_datasets=list(ds_cli) if ds_cli else None,
        multi_steps_mode=str(args.multi_steps_mode) if args.multi_steps_mode is not None else None,
        max_steps=int(args.max_steps) if args.max_steps is not None else None,
        multi_drop_last=args.multi_drop_last,
        rerank_enabled=args.rerank_enabled,
        rerank_top_k=int(args.rerank_top_k) if args.rerank_top_k is not None else None,
        rerank_epochs=int(args.rerank_epochs) if args.rerank_epochs is not None else None,
        hpo_baseline_epochs=int(args.hpo_baseline_epochs) if args.hpo_baseline_epochs is not None else None,
        hpo_grid_epochs=int(args.hpo_grid_epochs) if args.hpo_grid_epochs is not None else None,
        hpo_baseline_max_steps=int(args.hpo_baseline_max_steps) if args.hpo_baseline_max_steps is not None else None,
        hpo_grid_max_steps=int(args.hpo_grid_max_steps) if args.hpo_grid_max_steps is not None else None,
        hpo_rerank_max_steps=int(args.hpo_rerank_max_steps) if args.hpo_rerank_max_steps is not None else None,
        ours_r=int(args.ours_r) if args.ours_r is not None else None,
        ours_R=int(args.ours_R) if args.ours_R is not None else None,
        ablate_interp=bool(args.ablate_interp),
        history_enabled=bool(args.history_enabled),
    )
    if spec.scenario is None:
        if spec.multi_enabled is True:
            spec.scenario = "multi_task"
        elif spec.multi_enabled is False:
            spec.scenario = "single_task"

    spec.runs_group = _augment_runs_group_with_datasets(
        spec.runs_group,
        multi_enabled=spec.multi_enabled,
        multi_datasets=spec.multi_datasets,
    )

    if spec.rerank_top_k is not None and spec.rerank_top_k <= 0:
        raise ValueError("--rerank_top_k must be > 0")
    if spec.rerank_epochs is not None and spec.rerank_epochs <= 0:
        raise ValueError("--rerank_epochs must be > 0")
    if spec.hpo_baseline_epochs is not None and spec.hpo_baseline_epochs <= 0:
        raise ValueError("--hpo_baseline_epochs must be > 0")
    if spec.hpo_grid_epochs is not None and spec.hpo_grid_epochs <= 0:
        raise ValueError("--hpo_grid_epochs must be > 0")
    if spec.hpo_baseline_max_steps is not None and spec.hpo_baseline_max_steps <= 0:
        raise ValueError("--hpo_baseline_max_steps must be > 0")
    if spec.hpo_grid_max_steps is not None and spec.hpo_grid_max_steps <= 0:
        raise ValueError("--hpo_grid_max_steps must be > 0")
    if spec.hpo_rerank_max_steps is not None and spec.hpo_rerank_max_steps <= 0:
        raise ValueError("--hpo_rerank_max_steps must be > 0")
    if spec.max_steps is not None and spec.max_steps <= 0:
        raise ValueError("--max_steps must be > 0")
    if spec.epochs is not None and spec.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if spec.multi_enabled is True and (not spec.multi_datasets):
        raise ValueError("--multi_enabled=true requires --multi_datasets")
    if spec.multi_steps_mode == "max_steps" and spec.multi_enabled is True and spec.max_steps is None:
        raise ValueError("--multi_steps_mode=max_steps with --multi_enabled=true requires --max_steps")
    if bool(args.with_ablations) and spec.scenario != "multi_task":
        raise ValueError("--with_ablations is only supported when --scenario=multi_task")

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
    if spec.scenario is not None:
        common_sets_global.append(f"task.scenario={spec.scenario}")
        if spec.multi_enabled is None:
            implied_multi = spec.scenario in {"multi_task", "multi_dataset"}
            common_sets_global.append(f"task.multi.enabled={str(implied_multi).lower()}")
    if spec.multi_enabled is not None:
        common_sets_global.append(f"task.multi.enabled={str(spec.multi_enabled).lower()}")
    if spec.multi_datasets is not None:
        common_sets_global.append("task.multi.datasets=" + json.dumps(spec.multi_datasets))
    if spec.multi_steps_mode is not None:
        common_sets_global.append(f"train.multi.steps_mode={spec.multi_steps_mode}")
    if spec.max_steps is not None:
        common_sets_global.append(f"train.max_steps={spec.max_steps}")
    if spec.multi_drop_last is not None:
        common_sets_global.append(f"train.multi.drop_last={str(spec.multi_drop_last).lower()}")
    if spec.epochs is not None:
        common_sets_global.append(f"train.epochs={spec.epochs}")
    for s in list(args.set or []):
        ss = str(s).strip()
        if ss:
            common_sets_global.append(ss)

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
    # If CLI omits --trials, fall back to the default HPO tier (96).
    if spec.trials is not None:
        hpo_sets_only.append(f"hpo.budget.total_trials={spec.trials}")
    else:
        hpo_sets_only.append("hpo.budget.total_trials=96")
    if spec.rerank_enabled is not None:
        hpo_sets_only.append(f"hpo.grid.rerank.enabled={str(spec.rerank_enabled).lower()}")
    if spec.rerank_top_k is not None:
        hpo_sets_only.append(f"hpo.grid.rerank.top_k={spec.rerank_top_k}")
    if spec.rerank_epochs is not None:
        hpo_sets_only.append(f"hpo.grid.rerank.epochs={spec.rerank_epochs}")
    if spec.hpo_baseline_epochs is not None:
        hpo_sets_only.append(f"hpo.grid.baseline_search_epochs={spec.hpo_baseline_epochs}")
    if spec.hpo_grid_epochs is not None:
        hpo_sets_only.append(f"hpo.grid.grid_epochs={spec.hpo_grid_epochs}")
    if spec.hpo_baseline_max_steps is not None:
        hpo_sets_only.append(f"hpo.grid.baseline_search_max_steps={spec.hpo_baseline_max_steps}")
    if spec.hpo_grid_max_steps is not None:
        hpo_sets_only.append(f"hpo.grid.grid_max_steps={spec.hpo_grid_max_steps}")
    if spec.hpo_rerank_max_steps is not None:
        hpo_sets_only.append(f"hpo.grid.rerank.max_steps={spec.hpo_rerank_max_steps}")
    if bool(args.with_ablations):
        hpo_sets_only.append("hpo.ablations.enabled=true")

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
    if str(args.resume_final or "").strip():
        final_dir = Path(str(args.resume_final)).resolve()
        _check_resume_final_compat(final_dir=final_dir, expected_spec=spec, with_ablations=bool(args.with_ablations))
    else:
        final_ts = time.strftime("%Y%m%d-%H%M%S")
        final_budget_tag = _pipeline_budget_tag(spec)
        final_dir = runs_root / (
            f"final__{_slug(spec.runs_group)}__from_{dbg_ts}__{dbg_hash}__{_slug(final_budget_tag)}__{final_ts}"
        )

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
    if bool(args.with_ablations):
        cmd_final += ["--set", "final.ablations.enabled=true"]

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
                    "scenario": spec.scenario,
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "multi_enabled": spec.multi_enabled,
                    "multi_datasets": spec.multi_datasets,
                    "multi_steps_mode": spec.multi_steps_mode,
                    "max_steps": spec.max_steps,
                    "multi_drop_last": spec.multi_drop_last,
                    "epochs": spec.epochs,
                    "total_trials": spec.trials,
                    "rerank_enabled": spec.rerank_enabled,
                    "rerank_top_k": spec.rerank_top_k,
                    "rerank_epochs": spec.rerank_epochs,
                    "hpo_baseline_epochs": spec.hpo_baseline_epochs,
                    "hpo_grid_epochs": spec.hpo_grid_epochs,
                    "hpo_baseline_max_steps": spec.hpo_baseline_max_steps,
                    "hpo_grid_max_steps": spec.hpo_grid_max_steps,
                    "hpo_rerank_max_steps": spec.hpo_rerank_max_steps,
                    "ours_r": spec.ours_r,
                    "ours_R": spec.ours_R,
                    "final_epochs": spec.final_epochs,
                    "final_seeds": spec.final_seeds,
                    "ablate_interp": spec.ablate_interp,
                    "history_enabled": spec.history_enabled,
                    "with_ablations": bool(args.with_ablations),
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

    plot_diag_cmd = ["python", "scripts/plot_mechanism_diagnostics.py", str(trial_runs)]
    sh(plot_diag_cmd)

    if bool(args.with_ablations):
        plot_ablate_cmd = ["python", "scripts/plot_ablation_compare.py", str(trial_runs)]
        sh(plot_ablate_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
