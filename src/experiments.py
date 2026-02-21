from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple

import time
import json
import numpy as np
import pandas as pd

from .data_loader import InstanceData, load_instance

SolutionFn = Callable[[InstanceData, int], pd.DataFrame]


@dataclass
class EvalBreakdown:
    total: float
    skill_deficit: float
    excess_workload: float
    feasible: bool
    infeasibility_reason: Optional[str] = None


@dataclass
class ExperimentResult:
    instance_id: str
    method: str
    seed: int
    runtime_s: float
    objective: float
    feasible: bool
    breakdown: Dict[str, Any]

def _weights(info: Dict[str, Any]) -> Tuple[float, float]:
    w = info.get("weights", {})
    w_skill = float(w.get("S2_room_nurse_skill", 1.0))
    w_work = float(w.get("S4_nurse_excessive_workload", 1.0))
    return w_skill, w_work


def _standardize_solution(sol: pd.DataFrame) -> pd.DataFrame:
    required = {"global_shift", "room_id", "nurse_id"}
    missing = required - set(sol.columns)
    if missing:
        raise ValueError(f"Solução sem colunas obrigatórias: {missing}")

    out = sol.copy()
    out["global_shift"] = out["global_shift"].astype(int)
    out["room_id"] = out["room_id"].astype(str)
    out["nurse_id"] = out["nurse_id"].astype(str)
    out = out.drop_duplicates(subset=["global_shift", "room_id"], keep="last").reset_index(drop=True)
    return out


def evaluate_solution(inst: InstanceData, solution: pd.DataFrame) -> EvalBreakdown:

    sol = _standardize_solution(solution)

    occ = inst.occupied_room_shifts[["global_shift", "room_id", "total_room_workload", "max_skill_required"]].copy()
    occ["global_shift"] = occ["global_shift"].astype(int)
    occ["room_id"] = occ["room_id"].astype(str)

    merged = occ.merge(sol, on=["global_shift", "room_id"], how="left", validate="one_to_one")
    if merged["nurse_id"].isna().any():
        miss = int(merged["nurse_id"].isna().sum())
        return EvalBreakdown(
            total=float("inf"),
            skill_deficit=float("inf"),
            excess_workload=float("inf"),
            feasible=False,
            infeasibility_reason=f"Falha de cobertura: {miss} room-shifts sem enfermeiro.",
        )

    nurse = inst.nurse_shifts[["nurse_id", "global_shift", "skill_level", "max_load"]].copy()
    nurse["nurse_id"] = nurse["nurse_id"].astype(str)
    nurse["global_shift"] = nurse["global_shift"].astype(int)

    merged2 = merged.merge(
        nurse,
        on=["nurse_id", "global_shift"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_nurse"),
    )
    if merged2["skill_level"].isna().any():
        bad = merged2[merged2["skill_level"].isna()][["global_shift", "room_id", "nurse_id"]].head(5)
        return EvalBreakdown(
            total=float("inf"),
            skill_deficit=float("inf"),
            excess_workload=float("inf"),
            feasible=False,
            infeasibility_reason=(
                "Falha de disponibilidade: nurse_id alocado em global_shift onde não está escalado. "
                f"Exemplos:\n{bad.to_string(index=False)}"
            ),
        )

    deficit = (merged2["max_skill_required"] - merged2["skill_level"]).clip(lower=0)
    skill_deficit = float(deficit.sum())

    wl = (
        merged2.groupby(["nurse_id", "global_shift"], as_index=False)
        .agg(total_assigned_workload=("total_room_workload", "sum"),
             max_load=("max_load", "first"))
    )
    wl["excess"] = (wl["total_assigned_workload"] - wl["max_load"]).clip(lower=0)
    excess_workload = float(wl["excess"].sum())

    w_skill, w_work = _weights(inst.info)
    total = w_skill * skill_deficit + w_work * excess_workload

    return EvalBreakdown(
        total=float(total),
        skill_deficit=float(skill_deficit),
        excess_workload=float(excess_workload),
        feasible=True,
    )



def run_one(inst: InstanceData, method_name: str, solver_fn: SolutionFn, seed: int) -> ExperimentResult:
    t0 = time.perf_counter()
    sol = solver_fn(inst, seed)
    runtime = time.perf_counter() - t0

    breakdown = evaluate_solution(inst, sol)
    return ExperimentResult(
        instance_id=inst.instance_id,
        method=method_name,
        seed=int(seed),
        runtime_s=float(runtime),
        objective=float(breakdown.total),
        feasible=bool(breakdown.feasible),
        breakdown=asdict(breakdown),
    )


def run_suite(
    data_dir: str | Path,
    instance_ids: List[str],
    methods: Dict[str, SolutionFn],
    repeats: int = 5,
    base_seed: int = 0,
    out_dir: Optional[str | Path] = "results",
    load_extras: bool = False,
) -> pd.DataFrame:

    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for iid in instance_ids:
        inst = load_instance(data_dir=data_dir, instance_id=iid, load_extras=load_extras)

        for mname, fn in methods.items():
            for r in range(repeats):
                seed = base_seed + 1000 * r + hash((iid, mname)) % 997
                res = run_one(inst, mname, fn, seed=seed)
                rows.append(asdict(res))

    df = pd.DataFrame(rows)

    if out_path is not None:
        df.to_csv(out_path / "results.csv", index=False)

        with (out_path / "results.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return df

def greedy_baseline(inst: InstanceData, seed: int) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    occ = inst.occupied_room_shifts[["global_shift", "room_id"]].copy()
    nurse = inst.nurse_shifts[["global_shift", "nurse_id", "skill_level"]].copy()
    nurse = nurse.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    nurse = nurse.sort_values(["global_shift", "skill_level"], ascending=[True, False])

    best_by_shift = nurse.groupby("global_shift", as_index=False).first()[["global_shift", "nurse_id"]]

    sol = occ.merge(best_by_shift, on="global_shift", how="left")
    return sol[["global_shift", "room_id", "nurse_id"]]