from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import json
import pandas as pd


@dataclass(frozen=True)
class InstanceData:
    instance_id: str
    root: Path
    info: Dict[str, Any]

    nurse_shifts: pd.DataFrame
    occupied_room_shifts: pd.DataFrame

    rooms: Optional[pd.DataFrame] = None
    persons_in_rooms: Optional[pd.DataFrame] = None
    patient_assignment: Optional[pd.DataFrame] = None


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_instance(data_dir: str | Path, instance_id: str, load_extras: bool = True) -> InstanceData:
    data_dir = Path(data_dir)
    inst_root = data_dir / instance_id

    if not inst_root.exists():
        raise FileNotFoundError(f"Instância não encontrada: {inst_root}")

    info_path = inst_root / "instance_info.json"
    nurse_path = inst_root / "nurse_shifts.csv"
    occ_path = inst_root / "occupied_room_shifts.csv"

    for p in [info_path, nurse_path, occ_path]:
        if not p.exists():
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {p}")

    info = _read_json(info_path)
    nurse_shifts = _read_csv(nurse_path)
    occupied = _read_csv(occ_path)

    nurse_shifts["nurse_id"] = nurse_shifts["nurse_id"].astype(str)
    nurse_shifts["shift"] = nurse_shifts["shift"].astype(str)
    nurse_shifts["day"] = nurse_shifts["day"].astype(int)
    nurse_shifts["global_shift"] = nurse_shifts["global_shift"].astype(int)
    nurse_shifts["skill_level"] = nurse_shifts["skill_level"].astype(int)
    nurse_shifts["max_load"] = pd.to_numeric(nurse_shifts["max_load"], errors="raise")

    occupied["room_id"] = occupied["room_id"].astype(str)
    occupied["shift"] = occupied["shift"].astype(str)
    occupied["day"] = occupied["day"].astype(int)
    occupied["global_shift"] = occupied["global_shift"].astype(int)
    occupied["total_room_workload"] = pd.to_numeric(occupied["total_room_workload"], errors="raise")
    occupied["max_skill_required"] = occupied["max_skill_required"].astype(int)

    rooms = persons = patient = None
    if load_extras:
        rooms_path = inst_root / "rooms.csv"
        persons_path = inst_root / "persons_in_rooms.csv"
        patient_path = inst_root / "patient_assignment.csv"

        if rooms_path.exists():
            rooms = _read_csv(rooms_path)
            if "room_id" in rooms.columns:
                rooms["room_id"] = rooms["room_id"].astype(str)

        if persons_path.exists():
            persons = _read_csv(persons_path)
            for col in ["room_id", "shift", "person_id"]:
                if col in persons.columns:
                    persons[col] = persons[col].astype(str)
            for col in ["day", "global_shift"]:
                if col in persons.columns:
                    persons[col] = persons[col].astype(int)

        if patient_path.exists():
            patient = _read_csv(patient_path)
            if "room_id" in patient.columns:
                patient["room_id"] = patient["room_id"].astype(str)

    return InstanceData(
        instance_id=instance_id,
        root=inst_root,
        info=info,
        nurse_shifts=nurse_shifts,
        occupied_room_shifts=occupied,
        rooms=rooms,
        persons_in_rooms=persons,
        patient_assignment=patient,
    )