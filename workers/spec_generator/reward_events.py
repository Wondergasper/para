"""
Phase 4 reward event store.

Stores verifier rewards as JSONL so model-improvement logic can consume them
without requiring a database in the early Phase 4 build.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Iterator


@dataclass
class RewardEvent:
    source_code: str
    candidate_code: str
    func_name: str
    verifier: str
    reward: float
    score: float
    success: bool
    created_at: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        if not data["created_at"]:
            data["created_at"] = datetime.now(UTC).isoformat()
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "RewardEvent":
        return cls(
            source_code=data.get("source_code", ""),
            candidate_code=data.get("candidate_code", ""),
            func_name=data.get("func_name", ""),
            verifier=data.get("verifier", ""),
            reward=float(data.get("reward", 0.0)),
            score=float(data.get("score", 0.0)),
            success=bool(data.get("success", False)),
            created_at=data.get("created_at", ""),
        )


class RewardEventStore:
    def __init__(self, path: str = "data/rewards.jsonl"):
        self.path = path

    def append(self, event: RewardEvent) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")

    def iter_events(self) -> Iterator[RewardEvent]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield RewardEvent.from_dict(json.loads(line))
