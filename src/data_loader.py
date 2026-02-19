from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import json
from functools import lru_cache

@dataclass
class InstanceData:
    instance_id: str
    info: Dict[str, Any]
    rooms: pd.DataFrame
    person_in_rooms: pd.DataFrame
    patient_assignments: pd.DataFrame
    nurse_shirfts: pd.DataFrame
    occupied_room_shifts: pd.DataFrame

class HospitalDataLoader:
    REQUIRED_FILES = {
        "instance_info.json",
        "rooms.csv",
        "persons_in_rooms.csv",
        "patient_assignment.csv",
        "nurse_shifts.csv",
        "occupied_room_shifts.csv",
    }

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Pasta '{data_root}' não encontrada.")
        
        def list_instance(self) -> List[str]:
            instances = []
            for folder in self.data_root.iterdir():
                if folder.is_dir() and (folder / "instance_info.json").exists():
                    instances.append(folder.name)
            return sorted(instances)
        
    def _validate_instance(self, instance_id: str) -> Path:
        instance_path = self.data_root / instance_id

        if not instance_path.exists():
            raise FileNotFoundError(f"Instancia '{instance_id}' não encontrada.")
        
        missing = [
            f for f in self.REQUIRED_FILES
            if not (instance_path / f).exists()
        ]

        if missing:
            raise FileNotFoundError(
                f"Arquivos ausentes na instancia '{instance_id}': {missing}"
            )
        return instance_path
    
    @lru_cache(maxsize=16)
    def load_instance(self, instance_id: str) -> InstanceData:
        instance_path = self._validate_instance(instance_id)

        #json
        with open(instance_path / "instance_info.json", "r", encoding="utf-8") as f:
            info = json.load(f) 

        #csv
        rooms = pd.read_csv(instance_path / "rooms.csv", dtype={"room_id": "string"})

        persons_in_rooms = pd.read_csv(
            instance_path / "person_in_rooms.csv",
            dtype={"person_id": "string",
                    "room_id": "string",
                    "day": "int64",
                    "shift": "string",
                    "global_shift": "int64",
                    "workload": "int64",
                    "skill_required": "int64"
            },
        )

        patient_assignment = pd.read_csv(
            instance_path / "patient_assignment.csv",
            dtype={
                "patient_id": "string",
                "admission_day": "int64",
                "room_id": "string",
                "length_of_stay": "int64",
                "gender": "string",
                "age_group": "string",
            },
        )

        nurse_shifts = pd.read_csv(
            instance_path / "nurse_shifts.csv",
            dtype={
                "nurse_id": "string",
                "skill_level": "int64",
                "day": "int64",
                "shift": "string",
                "global_shift": "int64",
                "max_load": "int64",
            },
        )

        occupied_room_shifts = pd.read_csv(
            instance_path / "occupied_room_shifts.csv",
            dtype={
                "room_id": "string",
                "day": "int64",
                "shift": "string",
                "global_shift": "int64",
                "total_room_workload": "int64",
                "max_skill_required": "int64",
            },
        )

        shift_types = info.get("shift_types", ["early", "late", "night"])

        for df in [persons_in_rooms, nurse_shifts, occupied_room_shifts]:
            if "shift" in df.columns:
                df["shift"] = pd.Categorical(df["shift"], categories=shift_types)

        if "gender" in patient_assignment.columns:
            patient_assignment["gender"] = patient_assignment["gender"].astype("category")

        if "age_group" in patient_assignment.columns:
            patient_assignment["age_group"] = patient_assignment["age_group"].astype("category")

        return InstanceData(
            instance_id=instance_id,
            info=info,
            rooms=rooms,
            persons_in_rooms=persons_in_rooms,
            patient_assignment=patient_assignment,
            nurse_shifts=nurse_shifts,
            occupied_room_shifts=occupied_room_shifts,
        )