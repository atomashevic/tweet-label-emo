import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from config import LABEL_NORM


CONSENSUS_ITEMS_PATH = Path("samples_18_25_consensus_items.csv")
CHATGPT_PATH = Path("emotions_chatgpt.jsonl")
GEMMA_PATH = Path("emotions_gemma.jsonl")
OUTPUT_CSV_PATH = Path("samples_18_25_gemma_chatgpt_majority.csv")
REPORT_MD_PATH = Path("samples_18_25_gemma_chatgpt_majority.md")
METRICS_JSON_PATH = Path("samples_18_25_gemma_chatgpt_majority.json")


def read_consensus_items() -> list[dict[str, str]]:
    with CONSENSUS_ITEMS_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_label_file(path: Path) -> dict[str, str]:
    labels = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tweet_id = obj.get("id_str") or obj.get("id")
            labels[tweet_id] = LABEL_NORM.get(obj["emotion"], obj["emotion"])
    return labels


def cohen_kappa(xs: list[str], ys: list[str], labels: list[str]) -> tuple[float, float]:
    n = len(xs)
    observed = sum(a == b for a, b in zip(xs, ys)) / n
    counts_x = Counter(xs)
    counts_y = Counter(ys)
    expected = sum((counts_x.get(label, 0) / n) * (counts_y.get(label, 0) / n) for label in labels)
    kappa = (observed - expected) / (1 - expected) if (1 - expected) else math.nan
    return observed, kappa


def compare_to_consensus(rows: list[dict[str, str]], model_key: str, consensus_key: str) -> dict[str, object]:
    kept = [row for row in rows if row[consensus_key]]
    y_true = [row[consensus_key] for row in kept]
    y_pred = [row[model_key] for row in kept]
    labels = sorted(set(y_true) | set(y_pred))
    accuracy, kappa = cohen_kappa(y_true, y_pred, labels)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth][pred] += 1
    macro_recall = sum(
        confusion[label][label] / sum(confusion[label].values())
        for label in sorted(confusion)
    ) / len(confusion)
    top_errors = Counter((truth, pred) for truth, pred in zip(y_true, y_pred) if truth != pred)
    return {
        "n_items": len(kept),
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "macro_recall": macro_recall,
        "top_errors": [
            {"consensus_label": truth, "model_label": pred, "count": count}
            for (truth, pred), count in top_errors.most_common(15)
        ],
    }


def pairwise_model_agreement(rows: list[dict[str, str]], left_key: str, right_key: str) -> dict[str, float]:
    y_left = [row[left_key] for row in rows]
    y_right = [row[right_key] for row in rows]
    labels = sorted(set(y_left) | set(y_right))
    agreement, kappa = cohen_kappa(y_left, y_right, labels)
    return {"agreement": agreement, "cohen_kappa": kappa}


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    rows = read_consensus_items()
    chatgpt = read_label_file(CHATGPT_PATH)
    gemma = read_label_file(GEMMA_PATH)

    enriched_rows = []
    missing_chatgpt = 0
    missing_gemma = 0
    for row in rows:
        tweet_id = row["tweet_id"]
        chatgpt_label = chatgpt.get(tweet_id, "")
        gemma_label = gemma.get(tweet_id, "")
        missing_chatgpt += int(not chatgpt_label)
        missing_gemma += int(not gemma_label)
        enriched = dict(row)
        enriched["chatgpt_label"] = chatgpt_label
        enriched["gemma_label"] = gemma_label
        enriched["chatgpt_matches_majority"] = str(
            bool(row["strict_majority_label"]) and row["strict_majority_label"] == chatgpt_label
        ).lower()
        enriched["gemma_matches_majority"] = str(
            bool(row["strict_majority_label"]) and row["strict_majority_label"] == gemma_label
        ).lower()
        enriched["chatgpt_matches_unique_highest_frequency"] = str(
            bool(row["unique_highest_frequency_label"]) and row["unique_highest_frequency_label"] == chatgpt_label
        ).lower()
        enriched["gemma_matches_unique_highest_frequency"] = str(
            bool(row["unique_highest_frequency_label"]) and row["unique_highest_frequency_label"] == gemma_label
        ).lower()
        enriched_rows.append(enriched)

    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "tweet_pos",
                "tweet_id",
                "text",
                "n_raters",
                "top_vote_count",
                "top_labels",
                "exclude_due_to_ne_mogu_da_razumem_top",
                "strict_majority_label",
                "unique_highest_frequency_label",
                "chatgpt_label",
                "gemma_label",
                "chatgpt_matches_majority",
                "gemma_matches_majority",
                "chatgpt_matches_unique_highest_frequency",
                "gemma_matches_unique_highest_frequency",
                "label_counts_json",
            ],
        )
        writer.writeheader()
        for row in enriched_rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

    filtered_rows = [
        row
        for row in enriched_rows
        if row["exclude_due_to_ne_mogu_da_razumem_top"] == "false"
        and row["chatgpt_label"]
        and row["gemma_label"]
    ]

    majority_chatgpt = compare_to_consensus(filtered_rows, "chatgpt_label", "strict_majority_label")
    majority_gemma = compare_to_consensus(filtered_rows, "gemma_label", "strict_majority_label")
    plurality_chatgpt = compare_to_consensus(filtered_rows, "chatgpt_label", "unique_highest_frequency_label")
    plurality_gemma = compare_to_consensus(filtered_rows, "gemma_label", "unique_highest_frequency_label")

    majority_rows = [row for row in filtered_rows if row["strict_majority_label"]]
    plurality_rows = [row for row in filtered_rows if row["unique_highest_frequency_label"]]

    pairwise_majority = {
        "human_majority_vs_chatgpt": pairwise_model_agreement(majority_rows, "strict_majority_label", "chatgpt_label"),
        "human_majority_vs_gemma": pairwise_model_agreement(majority_rows, "strict_majority_label", "gemma_label"),
        "chatgpt_vs_gemma": pairwise_model_agreement(majority_rows, "chatgpt_label", "gemma_label"),
        "three_way_exact_agreement_rate": sum(
            row["strict_majority_label"] == row["chatgpt_label"] == row["gemma_label"]
            for row in majority_rows
        ) / len(majority_rows),
    }

    pairwise_plurality = {
        "human_unique_highest_frequency_vs_chatgpt": pairwise_model_agreement(plurality_rows, "unique_highest_frequency_label", "chatgpt_label"),
        "human_unique_highest_frequency_vs_gemma": pairwise_model_agreement(plurality_rows, "unique_highest_frequency_label", "gemma_label"),
        "chatgpt_vs_gemma": pairwise_model_agreement(plurality_rows, "chatgpt_label", "gemma_label"),
        "three_way_exact_agreement_rate": sum(
            row["unique_highest_frequency_label"] == row["chatgpt_label"] == row["gemma_label"]
            for row in plurality_rows
        ) / len(plurality_rows),
    }

    metrics = {
        "missing": {"chatgpt": missing_chatgpt, "gemma": missing_gemma},
        "majority": {
            "chatgpt": majority_chatgpt,
            "gemma": majority_gemma,
            "pairwise": pairwise_majority,
        },
        "unique_highest_frequency": {
            "chatgpt": plurality_chatgpt,
            "gemma": plurality_gemma,
            "pairwise": pairwise_plurality,
        },
    }
    with METRICS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    lines = [
        "# Samples 18-25: Human Majority, ChatGPT, and Gemma",
        "",
        f"- Missing ChatGPT labels in consensus table: {missing_chatgpt}",
        f"- Missing Gemma labels in consensus table: {missing_gemma}",
        "",
        "## Strict Majority Comparison",
        "",
        f"- ChatGPT items: {majority_chatgpt['n_items']}",
        f"- ChatGPT accuracy: {fmt(majority_chatgpt['accuracy'])}",
        f"- ChatGPT Cohen's kappa: {fmt(majority_chatgpt['cohen_kappa'])}",
        f"- Gemma items: {majority_gemma['n_items']}",
        f"- Gemma accuracy: {fmt(majority_gemma['accuracy'])}",
        f"- Gemma Cohen's kappa: {fmt(majority_gemma['cohen_kappa'])}",
        "",
        f"- Human majority vs ChatGPT kappa: {fmt(pairwise_majority['human_majority_vs_chatgpt']['cohen_kappa'])}",
        f"- Human majority vs Gemma kappa: {fmt(pairwise_majority['human_majority_vs_gemma']['cohen_kappa'])}",
        f"- ChatGPT vs Gemma kappa: {fmt(pairwise_majority['chatgpt_vs_gemma']['cohen_kappa'])}",
        f"- Three-way exact agreement rate: {fmt(pairwise_majority['three_way_exact_agreement_rate'])}",
        "",
        "## Unique Highest-Frequency Comparison",
        "",
        f"- ChatGPT items: {plurality_chatgpt['n_items']}",
        f"- ChatGPT accuracy: {fmt(plurality_chatgpt['accuracy'])}",
        f"- ChatGPT Cohen's kappa: {fmt(plurality_chatgpt['cohen_kappa'])}",
        f"- Gemma items: {plurality_gemma['n_items']}",
        f"- Gemma accuracy: {fmt(plurality_gemma['accuracy'])}",
        f"- Gemma Cohen's kappa: {fmt(plurality_gemma['cohen_kappa'])}",
        "",
        f"- Human plurality vs ChatGPT kappa: {fmt(pairwise_plurality['human_unique_highest_frequency_vs_chatgpt']['cohen_kappa'])}",
        f"- Human plurality vs Gemma kappa: {fmt(pairwise_plurality['human_unique_highest_frequency_vs_gemma']['cohen_kappa'])}",
        f"- ChatGPT vs Gemma kappa: {fmt(pairwise_plurality['chatgpt_vs_gemma']['cohen_kappa'])}",
        f"- Three-way exact agreement rate: {fmt(pairwise_plurality['three_way_exact_agreement_rate'])}",
        "",
    ]
    REPORT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV_PATH}")
    print(f"Wrote {REPORT_MD_PATH}")
    print(f"Wrote {METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
