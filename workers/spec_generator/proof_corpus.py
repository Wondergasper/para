"""
Phase 3 proof corpus store.

APG_System_Architecture.docx describes a PostgreSQL proof table in Phase 3.
This local JSONL store keeps the same shape without adding infrastructure yet.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Iterator


@dataclass
class ProofRecord:
    source_code: str
    parallel_code: str
    verifier: str
    success: bool
    score: float
    proof: str = ""
    annotated_ir: dict = field(default_factory=dict)
    dep_graph: dict = field(default_factory=dict)
    func_name: str = ""
    created_at: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        if not data["created_at"]:
            data["created_at"] = datetime.now(UTC).isoformat()
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "ProofRecord":
        return cls(
            source_code=data.get("source_code", ""),
            parallel_code=data.get("parallel_code", ""),
            verifier=data.get("verifier", ""),
            success=bool(data.get("success", False)),
            score=float(data.get("score", 0.0)),
            proof=data.get("proof", ""),
            annotated_ir=data.get("annotated_ir") or {},
            dep_graph=data.get("dep_graph") or {},
            func_name=data.get("func_name", ""),
            created_at=data.get("created_at", ""),
        )


class ProofCorpusStore:
    def __init__(self, path: str = "data/proofs.jsonl"):
        self.path = path

    def append(self, record: ProofRecord) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("source_code") == record.source_code and data.get("parallel_code") == record.parallel_code:
                                return
                        except json.JSONDecodeError:
                            continue
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")

    def iter_records(self) -> Iterator[ProofRecord]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield ProofRecord.from_dict(json.loads(line))

    def iter_verified(self, min_score: float = 0.8) -> Iterator[ProofRecord]:
        for record in self.iter_records():
            if record.success and record.score >= min_score:
                yield record
