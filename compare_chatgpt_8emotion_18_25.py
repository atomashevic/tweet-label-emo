import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from config import EMOTIONS, LABEL_NORM


CONSENSUS_ITEMS_PATH = Path("samples_18_25_consensus_items.csv")
CHATGPT_PATH = Path("emotions_chatgpt.jsonl")
REPORT_PATH = Path("samples_18_25_chatgpt_8emotion_report.md")
METRICS_PATH = Path("samples_18_25_chatgpt_8emotion_metrics.json")
CSV_PATH = Path("samples_18_25_chatgpt_8emotion_items.csv")

EIGHT_EMOTIONS = set(EMOTIONS)


def read_consensus_items() -> list[dict[str, str]]:
    with CONSENSUS_ITEMS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_chatgpt_labels() -> dict[str, str]:
    labels = {}
    with CHATGPT_PATH.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels[obj["id_str"]] = LABEL_NORM.get(obj["emotion"], obj["emotion"])
    return labels


def cohen_kappa(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, float]:
    n = len(y_true)
    observed = sum(a == b for a, b in zip(y_true, y_pred)) / n
    counts_true = Counter(y_true)
    counts_pred = Counter(y_pred)
    expected = sum(
        (counts_true.get(label, 0) / n) * (counts_pred.get(label, 0) / n)
        for label in labels
    )
    kappa = (observed - expected) / (1 - expected) if (1 - expected) else math.nan
    return observed, kappa


def evaluate(rows: list[dict[str, str]], consensus_key: str, chatgpt: dict[str, str]) -> dict[str, object]:
    kept = [
        row
        for row in rows
        if row[consensus_key] in EIGHT_EMOTIONS
    ]
    y_true = [row[consensus_key] for row in kept]
    y_pred = [chatgpt[row["tweet_id"]] for row in kept]
    labels = sorted(set(y_true) | set(y_pred))
    accuracy, kappa = cohen_kappa(y_true, y_pred, labels)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth][pred] += 1
    macro_recall = sum(
        confusion[label][label] / sum(confusion[label].values())
        for label in sorted(confusion)
    ) / len(confusion)
    errors = Counter((truth, pred) for truth, pred in zip(y_true, y_pred) if truth != pred)
    return {
        "n_items": len(kept),
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "macro_recall": macro_recall,
        "top_errors": [
            {"consensus_label": truth, "chatgpt_label": pred, "count": count}
            for (truth, pred), count in errors.most_common(15)
        ],
    }


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    rows = read_consensus_items()
    chatgpt = read_chatgpt_labels()

    filtered_rows = [
        row
        for row in rows
        if row["exclude_due_to_ne_mogu_da_razumem_top"] == "false"
    ]

    for row in filtered_rows:
        row["chatgpt_label"] = chatgpt[row["tweet_id"]]

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "tweet_pos",
                "tweet_id",
                "text",
                "strict_majority_label",
                "unique_highest_frequency_label",
                "chatgpt_label",
                "strict_majority_is_8emotion",
                "unique_highest_frequency_is_8emotion",
            ],
        )
        writer.writeheader()
        for row in filtered_rows:
            writer.writerow(
                {
                    "sample_idx": row["sample_idx"],
                    "tweet_pos": row["tweet_pos"],
                    "tweet_id": row["tweet_id"],
                    "text": row["text"],
                    "strict_majority_label": row["strict_majority_label"],
                    "unique_highest_frequency_label": row["unique_highest_frequency_label"],
                    "chatgpt_label": row["chatgpt_label"],
                    "strict_majority_is_8emotion": str(row["strict_majority_label"] in EIGHT_EMOTIONS).lower(),
                    "unique_highest_frequency_is_8emotion": str(row["unique_highest_frequency_label"] in EIGHT_EMOTIONS).lower(),
                }
            )

    overall_majority = evaluate(filtered_rows, "strict_majority_label", chatgpt)
    overall_plurality = evaluate(filtered_rows, "unique_highest_frequency_label", chatgpt)

    by_sample = {}
    for sample_idx in sorted({int(row["sample_idx"]) for row in filtered_rows}):
        sample_rows = [row for row in filtered_rows if int(row["sample_idx"]) == sample_idx]
        by_sample[sample_idx] = {
            "majority": evaluate(sample_rows, "strict_majority_label", chatgpt),
            "unique_highest_frequency": evaluate(sample_rows, "unique_highest_frequency_label", chatgpt),
        }

    payload = {
        "method": "Restrict comparison to items whose human consensus label is one of the 8 Plutchik emotions, so the target space matches ChatGPT's output space.",
        "overall": {
            "majority": overall_majority,
            "unique_highest_frequency": overall_plurality,
        },
        "by_sample": by_sample,
    }
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# ChatGPT vs Human Consensus in 8-Emotion Space",
        "",
        "Method: exclude items where the filtered human consensus label is not one of the 8 emotions, so ChatGPT is evaluated only on labels it could in principle output.",
        "",
        "## Overall",
        "",
        f"- Strict-majority 8-emotion items: {overall_majority['n_items']}",
        f"- Strict-majority accuracy: {fmt(overall_majority['accuracy'])}",
        f"- Strict-majority Cohen's kappa: {fmt(overall_majority['cohen_kappa'])}",
        f"- Strict-majority macro recall: {fmt(overall_majority['macro_recall'])}",
        "",
        f"- Unique-highest-frequency 8-emotion items: {overall_plurality['n_items']}",
        f"- Unique-highest-frequency accuracy: {fmt(overall_plurality['accuracy'])}",
        f"- Unique-highest-frequency Cohen's kappa: {fmt(overall_plurality['cohen_kappa'])}",
        f"- Unique-highest-frequency macro recall: {fmt(overall_plurality['macro_recall'])}",
        "",
        "## By Sample",
        "",
    ]

    for sample_idx in sorted(by_sample):
        majority = by_sample[sample_idx]["majority"]
        plurality = by_sample[sample_idx]["unique_highest_frequency"]
        lines.extend(
            [
                f"### Sample {sample_idx}",
                "",
                f"- Strict-majority items: {majority['n_items']}",
                f"- Strict-majority accuracy: {fmt(majority['accuracy'])}",
                f"- Strict-majority kappa: {fmt(majority['cohen_kappa'])}",
                f"- Unique-highest-frequency items: {plurality['n_items']}",
                f"- Unique-highest-frequency accuracy: {fmt(plurality['accuracy'])}",
                f"- Unique-highest-frequency kappa: {fmt(plurality['cohen_kappa'])}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
