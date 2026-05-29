import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from config import LABEL_NORM


SAMPLES = {18, 19, 20}
MAJORITY_PATH = Path("ff101_ff125_majority_vote.csv")
CHATGPT_PATH = Path("emotions_chatgpt.jsonl")
COMPARISON_CSV_PATH = Path("samples_18_20_chatgpt_vs_majority.csv")
REPORT_MD_PATH = Path("samples_18_20_chatgpt_vs_majority.md")
METRICS_JSON_PATH = Path("samples_18_20_chatgpt_vs_majority.json")


def load_majority_rows() -> list[dict[str, str]]:
    with MAJORITY_PATH.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if int(row["sample_idx"]) in SAMPLES]


def load_chatgpt_labels() -> dict[str, str]:
    labels = {}
    with CHATGPT_PATH.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels[obj["id_str"]] = LABEL_NORM.get(obj["emotion"], obj["emotion"])
    return labels


def cohen_kappa(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, float]:
    n = len(y_true)
    observed = sum(a == b for a, b in zip(y_true, y_pred)) / n
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    expected = sum(
        (true_counts.get(label, 0) / n) * (pred_counts.get(label, 0) / n)
        for label in labels
    )
    kappa = (observed - expected) / (1 - expected) if (1 - expected) else float("nan")
    return observed, kappa


def evaluate(rows: list[dict[str, str]], chatgpt: dict[str, str]) -> dict[str, object]:
    labels = sorted(
        set(row["majority_label"] for row in rows)
        | set(chatgpt[row["tweet_id"]] for row in rows)
    )
    y_true = [row["majority_label"] for row in rows]
    y_pred = [chatgpt[row["tweet_id"]] for row in rows]

    observed, kappa = cohen_kappa(y_true, y_pred, labels)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth][pred] += 1

    macro_recall = sum(
        confusion[label][label] / sum(confusion[label].values())
        for label in sorted(confusion)
    ) / len(confusion)

    errors = Counter(
        (truth, pred) for truth, pred in zip(y_true, y_pred) if truth != pred
    )

    return {
        "n_items": len(rows),
        "accuracy": observed,
        "cohen_kappa": kappa,
        "macro_recall": macro_recall,
        "majority_distribution": dict(Counter(y_true)),
        "chatgpt_distribution": dict(Counter(y_pred)),
        "top_errors": [
            {"majority_label": truth, "chatgpt_label": pred, "count": count}
            for (truth, pred), count in errors.most_common(15)
        ],
    }


def write_comparison_csv(rows: list[dict[str, str]], chatgpt: dict[str, str]) -> None:
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "sample_idx": row["sample_idx"],
                "tweet_pos": row["tweet_pos"],
                "tweet_id": row["tweet_id"],
                "text": row["text"],
                "majority_label": row["majority_label"],
                "majority_vote_count": row["majority_vote_count"],
                "majority_vote_share": row["majority_vote_share"],
                "is_tie": row["is_tie"],
                "tied_labels": row["tied_labels"],
                "chatgpt_label": chatgpt[row["tweet_id"]],
                "matches_majority": str(chatgpt[row["tweet_id"]] == row["majority_label"]).lower(),
            }
        )

    out_rows.sort(key=lambda row: (int(row["sample_idx"]), int(row["tweet_pos"])))
    with COMPARISON_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "tweet_pos",
                "tweet_id",
                "text",
                "majority_label",
                "majority_vote_count",
                "majority_vote_share",
                "is_tie",
                "tied_labels",
                "chatgpt_label",
                "matches_majority",
            ],
        )
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)


def write_report(
    strict_majority_metrics: dict[str, object],
    strict_majority_non_ncdr_metrics: dict[str, object],
) -> None:
    lines = [
        "# ChatGPT vs Human Majority Labels on Samples 18-20",
        "",
        "## Strict Majority",
        "",
        f"- Items: {strict_majority_metrics['n_items']}",
        f"- Accuracy: {strict_majority_metrics['accuracy']:.4f}",
        f"- Cohen's kappa: {strict_majority_metrics['cohen_kappa']:.4f}",
        f"- Macro recall: {strict_majority_metrics['macro_recall']:.4f}",
        "",
        "Top errors:",
    ]
    for error in strict_majority_metrics["top_errors"]:
        lines.append(
            f"- {error['majority_label']} -> {error['chatgpt_label']}: {error['count']}"
        )

    lines.extend(
        [
            "",
            "## Strict Majority Excluding `Ne mogu da razumem`",
            "",
            f"- Items: {strict_majority_non_ncdr_metrics['n_items']}",
            f"- Accuracy: {strict_majority_non_ncdr_metrics['accuracy']:.4f}",
            f"- Cohen's kappa: {strict_majority_non_ncdr_metrics['cohen_kappa']:.4f}",
            f"- Macro recall: {strict_majority_non_ncdr_metrics['macro_recall']:.4f}",
            "",
            "Top errors:",
        ]
    )
    for error in strict_majority_non_ncdr_metrics["top_errors"]:
        lines.append(
            f"- {error['majority_label']} -> {error['chatgpt_label']}: {error['count']}"
        )

    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    majority_rows = load_majority_rows()
    chatgpt = load_chatgpt_labels()

    strict_majority_rows = [
        row for row in majority_rows if row["is_tie"] == "false"
    ]
    strict_majority_non_ncdr_rows = [
        row
        for row in strict_majority_rows
        if row["majority_label"] != "Ne mogu da razumem"
    ]

    strict_majority_metrics = evaluate(strict_majority_rows, chatgpt)
    strict_majority_non_ncdr_metrics = evaluate(strict_majority_non_ncdr_rows, chatgpt)

    write_comparison_csv(majority_rows, chatgpt)
    write_report(strict_majority_metrics, strict_majority_non_ncdr_metrics)
    with METRICS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strict_majority": strict_majority_metrics,
                "strict_majority_non_ncdr": strict_majority_non_ncdr_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {COMPARISON_CSV_PATH}")
    print(f"Wrote {REPORT_MD_PATH}")
    print(f"Wrote {METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
