"""Exact NumPy comparison, decision-preserving feedback, and the private loop."""
import numpy as np

from src.common import normalize


def compare(vector, gallery):
    scores = np.clip(gallery @ vector, -1.0, 1.0)
    position = int(np.argmax(scores))  # Stable first enrollment wins an exact tie.
    return float(scores[position]), position


def feedback(score, granularity, threshold, lo):
    score = float(np.clip(score, -1.0, 1.0))
    if granularity == "float":
        return score
    if granularity == "1":
        return int(score >= threshold)
    if granularity not in ("8", "4"):
        raise ValueError("Feedback must be float, 8, 4, or 1.")
    if not -1 <= lo < threshold <= 1:
        raise ValueError("Quantization requires -1 <= lo < threshold <= 1.")
    half = 2 ** (int(granularity) - 1)
    if score < threshold:
        fraction = (score - lo) / (threshold - lo)
        return int(np.clip(np.floor(half * fraction), 0, half - 1))
    if threshold == 1:
        return half
    fraction = (score - threshold) / (1 - threshold)
    return half + int(np.clip(np.floor(half * fraction), 0, half - 1))


def recovery(vector, identity, held_out, recovery_threshold):
    scores = np.clip(held_out[identity] @ vector, -1, 1)
    median = float(np.median(scores))
    return median, bool(median >= recovery_threshold)


def run_loop(candidates, gallery, identities, held_out, threshold, lo,
             recovery_threshold, granularity, budget):
    """A generator sees only a scalar response to each rejected query.

    The evaluator owns identities and held-out data. It closes the generator
    immediately at first acceptance, before any private result could be sent.
    """
    if budget < 1:
        raise ValueError("Query budget must be positive.")
    record = {"status": "budget_exhausted", "queries": 0, "accepted": False,
              "acceptance_query": None, "first_acceptance_score": None,
              "matched_identity": None, "held_out_median": None, "recovered": False}
    started = False
    visible = None
    try:
        while record["queries"] < budget:
            try:
                candidate = candidates.send(visible) if started else next(candidates)
                started = True
            except StopIteration:
                record["status"] = "pool_exhausted"
                break
            record["queries"] += 1
            try:
                vector = normalize(candidate)
            except ValueError as exc:
                record.update(status="invalid_candidate", error=str(exc))
                break
            score, match = compare(vector, gallery)
            visible = feedback(score, granularity, threshold, lo)
            if score >= threshold:
                identity = identities[match]
                median, recovered = recovery(vector, identity, held_out, recovery_threshold)
                record.update(status="accepted", accepted=True,
                              acceptance_query=record["queries"], first_acceptance_score=score,
                              matched_identity=identity, held_out_median=median, recovered=recovered)
                break
    finally:
        candidates.close()
    return record
