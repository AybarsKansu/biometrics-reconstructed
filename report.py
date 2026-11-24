"""Derive tables and plots from recorded experiments or H histogram counts."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_config, read_json, write_csv


def success_quantile(records, outcome, fraction):
    successful = sorted(r["acceptance_query"] for r in records if r[outcome])
    rank = int(np.ceil(fraction * len(records)))
    if len(successful) >= rank:
        return successful[rank - 1]
    if all(r["status"] in ("accepted", "budget_exhausted") for r in records):
        return f"> {records[0]['budget']} / not reached"
    return f"not reached (budget {records[0]['budget']}; includes early exhaustion or blocked runs)"


def summarize(directory):
    directory = Path(directory)
    snapshot = read_json(directory / "resolved.json")
    records = [json.loads(line) for line in (directory / "private_records.jsonl").read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("No experiment records.")
    budgets = {r["budget"] for r in records}
    if len(budgets) != 1:
        raise ValueError("Do not combine unequal budgets in a batch report.")
    budget = budgets.pop()
    groups = defaultdict(list)
    for record in records:
        groups[record["target_fmr"], record["granularity"], record["method"]].append(record)
    summary = []
    for (target, granularity, method), items in sorted(groups.items()):
        accepted = sum(r["accepted"] for r in items)
        recovered = sum(r["recovered"] for r in items)
        summary.append({"target_fmr": target, "granularity": granularity, "method": method,
                        "runs": len(items), "budget": budget, "accepted": accepted, "recovered": recovered,
                        "acceptance_rate_all_runs": accepted / len(items),
                        "recovery_rate_all_runs": recovered / len(items),
                        "recovery_rate_accepted_runs": recovered / accepted if accepted else "undefined",
                        "acceptance_median_query_all_runs": success_quantile(items, "accepted", 0.5),
                        "acceptance_p90_query_all_runs": success_quantile(items, "accepted", 0.9),
                        "recovery_median_query_all_runs": success_quantile(items, "recovered", 0.5),
                        "recovery_p90_query_all_runs": success_quantile(items, "recovered", 0.9),
                        "budget_exhausted": sum(r["status"] == "budget_exhausted" for r in items),
                        "other_unsuccessful": sum(not r["accepted"] and r["status"] != "budget_exhausted" for r in items)})
    write_csv(directory / "summary.csv", summary)
    queries = np.arange(budget + 1)
    curve_rows = []
    for target in snapshot["targets"]:
        for granularity in snapshot["granularities"]:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for method in ["zero"] + snapshot["methods"]:
                items = groups.get((target, "independent" if method == "zero" else granularity, method), [])
                if not items:
                    continue
                for ax, outcome in zip(axes, ("accepted", "recovered")):
                    success = np.array([r["acceptance_query"] if r[outcome] else np.inf for r in items])
                    rates = np.mean(success[:, None] <= queries[None, :], axis=0)
                    ax.step(queries, rates, where="post", label=method)
                    curve_rows.extend({"target_fmr": target, "granularity": granularity, "method": method,
                                       "outcome": outcome, "query": int(q), "rate_all_runs": float(rate)}
                                      for q, rate in zip(queries, rates))
                    ax.set(xlabel="Oracle queries", title="Acceptance" if outcome == "accepted" else "Recovery at first acceptance",
                           ylim=(0, 1.03), xlim=(0, budget))
                    ax.grid(alpha=0.25)
            axes[0].set_ylabel("Fraction of all runs")
            axes[1].legend(loc="best")
            feedback_label = "float" if granularity == "float" else granularity + "-bit"
            fig.suptitle(f"{snapshot['label']} | FMR {target:g} | {feedback_label} feedback | budget {budget}")
            fig.tight_layout()
            fig.savefig(directory / f"success_fmr-{target}_{granularity}.png", dpi=150)
            plt.close(fig)
    write_csv(directory / "success_curves.csv", curve_rows)
    frequent = [r for r in summary if r["budget_exhausted"] / r["runs"] >= 0.5]
    note = ["All rates retain unsuccessful runs. Recovery uses the same first accepted vector.",
            "Overall query quantiles include all runs; unreached quantiles are not successful-only summaries.",
            "Few repetitions, correlated starts and calibration scores limit comparisons. No confidence intervals."]
    if frequent:
        note.append("At least half the runs exhausted the budget in these measured settings; consider a separately labeled larger-budget pilot after reviewing them:")
        note.extend(f"{r['method']}/{r['granularity']}/FMR={r['target_fmr']}: {r['budget_exhausted']}/{r['runs']} exhausted {budget} queries" for r in frequent)
    (directory / "interpretation.txt").write_text("\n".join(note) + "\n", encoding="utf-8")
    print(f"Derived summary, curves, and plots in {directory}")
    return summary


def h_report(config):
    with Path(config["paths"]["h_histogram"]).open(encoding="utf-8", newline="") as stream:
        hist_rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in hist_rows:
        groups[row["kind"]].append(row)
    destination = Path(config["paths"]["results"]) / "h-calibration"
    destination.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row_number, protocol in enumerate(("system", "pair")):
        impostor = groups[protocol + "_impostor"]
        genuine = groups[protocol + "_genuine"]
        thresholds = np.array([float(row["lower"]) for row in impostor])
        ic = np.array([int(row["count"]) for row in impostor])
        gc = np.array([int(row["count"]) for row in genuine])
        if ic.sum() == 0 or gc.sum() == 0:
            raise ValueError("H needs genuine and impostor scores for DET/ROC.")
        fmr = np.cumsum(ic[::-1])[::-1] / ic.sum()
        tpr = np.cumsum(gc[::-1])[::-1] / gc.sum()
        fnmr = 1 - tpr
        curve_rows.extend({"protocol": protocol, "threshold": float(t), "fmr": float(f),
                           "fnmr": float(n), "tpr": float(p), "basis": "binned_H_counts"}
                          for t, f, n, p in zip(thresholds, fmr, fnmr, tpr))
        axes[row_number, 0].plot(thresholds, ic / ic.sum(), label="impostor")
        axes[row_number, 0].plot(thresholds, gc / gc.sum(), label="genuine")
        axes[row_number, 0].set(xlabel="Cosine (bin lower edge)", ylabel="Fraction per bin", title=protocol + " scores")
        axes[row_number, 0].legend()
        axes[row_number, 1].plot(fmr, tpr)
        axes[row_number, 1].set(xlabel="FMR", ylabel="True match rate", title="Binned ROC", xlim=(0, 1), ylim=(0, 1))
        normal = NormalDist()
        det_x = [normal.inv_cdf(float(p)) for p in np.clip(fmr, 0.5 / ic.sum(), 1 - 0.5 / ic.sum())]
        det_y = [normal.inv_cdf(float(p)) for p in np.clip(fnmr, 0.5 / gc.sum(), 1 - 0.5 / gc.sum())]
        axes[row_number, 2].plot(det_x, det_y)
        axes[row_number, 2].set(xlabel="Normal quantile of FMR", ylabel="Normal quantile of FNMR", title="Binned DET (endpoints clipped for display)")
    fig.suptitle("LFW H calibration score diagnostics")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(destination / "h_distributions_roc_det.png", dpi=150)
    plt.close(fig)
    write_csv(destination / "h_roc_det.csv", curve_rows)
    print(destination)


if __name__ == "__main__":
    config = load_config()
    if Path(config["paths"]["h_histogram"]).exists():
        h_report(config)
    results = Path(config["paths"]["results"])
    batches = sorted(path.parent for path in results.glob("*/private_records.jsonl"))
    if batches:
        summarize(batches[-1])
    else:
        print("No experiment records yet. Run experiment.py first.")
