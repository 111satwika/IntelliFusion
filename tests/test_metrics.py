"""Tests for app.evaluation.metrics's deterministic retrieval metrics
(precision_at_k / recall_at_k). Uses the same dict-based fake-chunk
shape as test_retriever.py (metadata.document_id / chunk_index)."""

from app.evaluation.metrics import precision_at_k, recall_at_k


def _chunk(document_id: str, chunk_index: int = 0) -> dict:
    return {"content": "x", "metadata": {"document_id": document_id, "chunk_index": chunk_index}}


def test_precision_at_k_no_expected_ids_returns_zero():
    retrieved = [_chunk("a")]
    assert precision_at_k(retrieved, [], mode="document") == 0.0


def test_precision_at_k_empty_retrieved_returns_zero():
    assert precision_at_k([], ["a"], mode="document") == 0.0


def test_precision_at_k_document_mode():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]
    # 2 of 3 retrieved docs are in the expected set.
    assert precision_at_k(retrieved, ["a", "c", "z"], mode="document") == 2 / 3


def test_precision_at_k_respects_k():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]
    # Only look at the top-1: "a" is expected -> precision 1.0.
    assert precision_at_k(retrieved, ["a"], mode="document", k=1) == 1.0
    # Top-2 slice ("a","b") has 1 of 2 expected.
    assert precision_at_k(retrieved, ["a"], mode="document", k=2) == 0.5


def test_precision_at_k_chunk_mode():
    retrieved = [_chunk("a", 0), _chunk("a", 1)]
    expected = ["a::0"]
    assert precision_at_k(retrieved, expected, mode="chunk") == 0.5


def test_recall_at_k_no_expected_ids_returns_zero():
    assert recall_at_k([_chunk("a")], [], mode="document") == 0.0


def test_recall_at_k_document_mode():
    retrieved = [_chunk("a"), _chunk("b")]
    # Only "a" of the two expected docs was retrieved -> recall 0.5.
    assert recall_at_k(retrieved, ["a", "z"], mode="document") == 0.5


def test_recall_at_k_all_expected_found_is_one():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]
    assert recall_at_k(retrieved, ["a", "b"], mode="document") == 1.0


def test_recall_at_k_respects_k():
    retrieved = [_chunk("a"), _chunk("b")]
    # "b" (the expected doc) is outside the top-1 slice.
    assert recall_at_k(retrieved, ["b"], mode="document", k=1) == 0.0
    assert recall_at_k(retrieved, ["b"], mode="document", k=2) == 1.0


def test_recall_at_k_chunk_mode():
    retrieved = [_chunk("a", 0)]
    expected = ["a::0", "a::1"]
    assert recall_at_k(retrieved, expected, mode="chunk") == 0.5
