import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from config import LABEL_NORM


ANNOTATIONS_PATH = Path("ff_samples_18_22_annotations.csv")
CHATGPT_PATH = Path("emotions_chatgpt.jsonl")
ITEMS_CSV_PATH = Path("samples_18_22_chatgpt_ignore_dontknow_items.csv")
REPORT_MD_PATH = Path("samples_18_22_chatgpt_ignore_dontknow_report.md")
METRICS_JSON_PATH = Path("samples_18_22_chatgpt_ignore_dontknow_metrics.json")

SAMPLES = [18, 19, 20, 21, 22]
DROP_LABEL = "Ne mogu da razumem"


def read_annotations() -> list[dict[str, str]]:
    with ANNOTATIONS_PATH.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if int(row["sample_idx"]) in SAMPLES]


def read_chatgpt() -> dict[str, str]:
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
    kappa = (observed - expected) / (1 - expected) if (1 - expected) else math.nan
    return observed, kappa


def group_items(rows: list[dict[str, str]]) -> dict[int, dict[str, list[dict[str, str]]]]:
    grouped: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["sample_idx"])][row["tweet_id"]].append(row)
    return grouped


def item_record(
    sample_idx: int,
    tweet_id: str,
    rows: list[dict[str, str]],
    chatgpt_labels: dict[str, str],
) -> dict[str, object]:
    valid_rows = [row for row in rows if row["label"] != DROP_LABEL]
    counts = Counter(row["label"] for row in valid_rows)
    n_valid = sum(counts.values())
    top_count = max(counts.values()) if counts else 0
    top_labels = sorted(label for label, count in counts.items() if count == top_count)
    unique_highest_frequency = top_labels[0] if len(top_labels) == 1 else ""
    strict_majority = (
        unique_highest_frequency
        if unique_highest_frequency and top_count > n_valid / 2
        else ""
    )
    tweet_pos = min(int(row["tweet_pos"]) for row in rows)
    chatgpt_label = chatgpt_labels.get(tweet_id, "")

    return {
        "sample_idx": sample_idx,
        "tweet_pos": tweet_pos,
        "tweet_id": tweet_id,
        "text": rows[0]["text"],
        "n_total_votes": len(rows),
        "n_dontknow_votes": len(rows) - n_valid,
        "n_valid_votes": n_valid,
        "top_vote_count_no_dontknow": top_count,
        "top_labels_no_dontknow": " | ".join(top_labels),
        "strict_majority_label_no_dontknow": strict_majority,
        "unique_highest_frequency_label_no_dontknow": unique_highest_frequency,
        "chatgpt_label": chatgpt_label,
        "matches_majority_no_dontknow": str(
            bool(strict_majority) and strict_majority == chatgpt_label
        ).lower(),
        "matches_unique_highest_frequency_no_dontknow": str(
            bool(unique_highest_frequency) and unique_highest_frequency == chatgpt_label
        ).lower(),
        "label_counts_no_dontknow_json": json.dumps(
            dict(sorted(counts.items())),
            ensure_ascii=False,
        ),
    }


def evaluate(records: list[dict[str, object]], label_key: str) -> dict[str, object]:
    rows = [record for record in records if record[label_key]]
    y_true = [record[label_key] for record in rows]
    y_pred = [record["chatgpt_label"] for record in rows]
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
        "n_items": len(rows),
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
    annotations = read_annotations()
    chatgpt = read_chatgpt()
    grouped = group_items(annotations)

    item_records = []
    for sample_idx in SAMPLES:
        for tweet_id, rows in sorted(grouped[sample_idx].items()):
            item_records.append(item_record(sample_idx, tweet_id, rows, chatgpt))
    item_records.sort(key=lambda row: (row["sample_idx"], row["tweet_pos"]))

    with ITEMS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "tweet_pos",
                "tweet_id",
                "text",
                "n_total_votes",
                "n_dontknow_votes",
                "n_valid_votes",
                "top_vote_count_no_dontknow",
                "top_labels_no_dontknow",
                "strict_majority_label_no_dontknow",
                "unique_highest_frequency_label_no_dontknow",
                "chatgpt_label",
                "matches_majority_no_dontknow",
                "matches_unique_highest_frequency_no_dontknow",
                "label_counts_no_dontknow_json",
            ],
        )
        writer.writeheader()
        for record in item_records:
            writer.writerow(record)

    eligible_records = [record for record in item_records if record["n_valid_votes"] >= 2]
    overall_majority = evaluate(eligible_records, "strict_majority_label_no_dontknow")
    overall_plurality = evaluate(
        eligible_records,
        "unique_highest_frequency_label_no_dontknow",
    )

    by_sample = {}
    for sample_idx in SAMPLES:
        sample_records = [
            record
            for record in eligible_records
            if record["sample_idx"] == sample_idx
        ]
        by_sample[sample_idx] = {
            "eligible_items": len(sample_records),
            "majority": evaluate(sample_records, "strict_majority_label_no_dontknow"),
            "unique_highest_frequency": evaluate(
                sample_records,
                "unique_highest_frequency_label_no_dontknow",
            ),
        }

    payload = {
        "method": "Ignore only human `Ne mogu da razumem` votes, require at least 2 remaining valid human votes per item, then compare ChatGPT to consensus on the remaining labels.",
        "overall": {
            "eligible_items": len(eligible_records),
            "majority": overall_majority,
            "unique_highest_frequency": overall_plurality,
        },
        "by_sample": by_sample,
    }
    with METRICS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# ChatGPT Comparison Ignoring `Ne mogu da razumem`",
        "",
        "Method: drop only the human `Ne mogu da razumem` votes, require at least 2 remaining valid human votes per item, and compare ChatGPT to consensus on the remaining labels.",
        "",
        "## Overall",
        "",
        f"- Eligible items: {len(eligible_records)}",
        f"- Majority items: {overall_majority['n_items']}",
        f"- Majority accuracy: {fmt(overall_majority['accuracy'])}",
        f"- Majority Cohen's kappa: {fmt(overall_majority['cohen_kappa'])}",
        f"- Majority macro recall: {fmt(overall_majority['macro_recall'])}",
        "",
        f"- Unique-highest-frequency items: {overall_plurality['n_items']}",
        f"- Unique-highest-frequency accuracy: {fmt(overall_plurality['accuracy'])}",
        f"- Unique-highest-frequency Cohen's kappa: {fmt(overall_plurality['cohen_kappa'])}",
        f"- Unique-highest-frequency macro recall: {fmt(overall_plurality['macro_recall'])}",
        "",
        "## By Sample",
        "",
    ]

    for sample_idx in SAMPLES:
        sample_payload = by_sample[sample_idx]
        majority = sample_payload["majority"]
        plurality = sample_payload["unique_highest_frequency"]
        lines.extend(
            [
                f"### Sample {sample_idx}",
                "",
                f"- Eligible items: {sample_payload['eligible_items']}",
                f"- Majority items: {majority['n_items']}",
                f"- Majority accuracy: {fmt(majority['accuracy'])}",
                f"- Majority Cohen's kappa: {fmt(majority['cohen_kappa'])}",
                f"- Unique-highest-frequency items: {plurality['n_items']}",
                f"- Unique-highest-frequency accuracy: {fmt(plurality['accuracy'])}",
                f"- Unique-highest-frequency Cohen's kappa: {fmt(plurality['cohen_kappa'])}",
                "",
            ]
        )

    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {ITEMS_CSV_PATH}")
    print(f"Wrote {REPORT_MD_PATH}")
    print(f"Wrote {METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
