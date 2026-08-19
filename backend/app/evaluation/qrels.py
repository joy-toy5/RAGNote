"""将 source evidence anchor 编译为指定索引版本的 chunk qrels。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from app.evaluation.contracts import sha256_json
from app.evaluation.dataset import (
    ChunkRecord,
    EvaluationDataset,
    EvidenceAnchor,
    dataset_fingerprint,
)


class QrelsCompilationError(ValueError):
    """source qrel 无法严格解析到稳定 chunk。"""


@dataclass(frozen=True, slots=True)
class CompiledQrel:
    query_id: str
    anchor_id: str
    evidence_set_id: str | None
    relevance: int
    chunk_ids: tuple[str, ...]
    document_id: str
    document_revision: int
    index_version: str


@dataclass(frozen=True, slots=True)
class CompiledQrels:
    schema_version: int
    qrels_version: str
    dataset_sha256: str
    index_version: str
    entries: tuple[CompiledQrel, ...]

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._payload(include_fingerprint=False))

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_fingerprint=True)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def _payload(self, *, include_fingerprint: bool) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "qrels_version": self.qrels_version,
            "dataset_sha256": self.dataset_sha256,
            "index_version": self.index_version,
            "entries": [asdict(entry) for entry in self.entries],
        }
        if include_fingerprint:
            payload["qrels_sha256"] = self.fingerprint
        return payload


def compile_qrels(dataset: EvaluationDataset) -> CompiledQrels:
    anchors = {anchor.anchor_id: anchor for anchor in dataset.anchors}
    entries = []
    for source_qrel in dataset.source_qrels:
        anchor = anchors[source_qrel.anchor_id]
        matches = _matching_chunks(dataset.chunks, anchor, dataset.index_version)
        if not matches:
            raise QrelsCompilationError(
                f"UNRESOLVED_ANCHOR: {anchor.anchor_id}"
            )
        entries.append(
            CompiledQrel(
                query_id=source_qrel.query_id,
                anchor_id=anchor.anchor_id,
                evidence_set_id=source_qrel.evidence_set_id,
                relevance=source_qrel.relevance,
                chunk_ids=tuple(chunk.chunk_id for chunk in matches),
                document_id=anchor.document_id,
                document_revision=anchor.document_revision,
                index_version=dataset.index_version,
            )
        )
    ordered_entries = tuple(
        sorted(entries, key=lambda entry: (entry.query_id, entry.anchor_id))
    )
    return CompiledQrels(
        schema_version=2,
        qrels_version=dataset.qrels_version,
        dataset_sha256=dataset_fingerprint(dataset),
        index_version=dataset.index_version,
        entries=ordered_entries,
    )


def _matching_chunks(
    chunks: tuple[ChunkRecord, ...],
    anchor: EvidenceAnchor,
    index_version: str,
) -> tuple[ChunkRecord, ...]:
    return tuple(
        sorted(
            (
                chunk
                for chunk in chunks
                if chunk.user_id == anchor.user_id
                and chunk.document_id == anchor.document_id
                and chunk.document_revision == anchor.document_revision
                and chunk.index_version == index_version
                and _page_matches(chunk.page_number, anchor.page_number)
                and chunk.char_start <= anchor.char_start
                and chunk.char_end >= anchor.char_end
            ),
            key=lambda chunk: chunk.chunk_id,
        )
    )


def _page_matches(chunk_page: int | None, anchor_page: int | None) -> bool:
    return anchor_page is None or chunk_page == anchor_page
