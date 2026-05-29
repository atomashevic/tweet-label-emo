import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from config import EMOTIONS, LABEL_NORM, SPECIAL_LABELS


ANNOTATIONS_PATH = Path("ff_samples_18_25_annotations.csv")
CHATGPT_PATH = Path("emotions_chatgpt.jsonl")
ITEMS_CSV_PATH = Path("samples_18_25_consensus_items.csv")
REPORT_MD_PATH = Path("samples_18_25_consensus_report.md")
METRICS_JSON_PATH = Path("samples_18_25_consensus_metrics.json")

SAMPLES = list(range(18, 26))
EXCLUDE_LABEL = "Ne mogu da razumem"
CATEGORIES = EMOTIONS + SPECIAL_LABELS


def read_annotations() -> list[dict[str, str]]:
    with ANNOTATIONS_PATH.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def read_chatgpt() -> dict[str, str]:
    labels = {}
    with CHATGPT_PATH.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels[obj["id_str"]] = LABEL_NORM.get(obj["emotion"], obj["emotion"])
    return labels


def label_counts(rows: list[dict[str, str]]) -> Counter[str]:
    counts = Counter(row["label"] for row in rows)
    unknown = sorted(label for label in counts if label not in CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")
    return counts


def item_record(item_rows: list[dict[str, str]], chatgpt_labels: dict[str, str]) -> dict[str, object]:
    counts = label_counts(item_rows)
    n_raters = len(item_rows)
    max_count = max(counts.values())
    top_labels = sorted(label for label, count in counts.items() if count == max_count)
    unique_plurality = top_labels[0] if len(top_labels) == 1 else ""
    strict_majority = unique_plurality if unique_plurality and max_count > n_raters / 2 else ""
    exclude_item = EXCLUDE_LABEL in top_labels
    tweet_pos = min(int(row["tweet_pos"]) for row in item_rows)
    tweet_id = item_rows[0]["tweet_id"]
    chatgpt_label = chatgpt_labels.get(tweet_id, "")

    return {
        "sample_idx": int(item_rows[0]["sample_idx"]),
        "tweet_pos": tweet_pos,
        "tweet_id": tweet_id,
        "text": item_rows[0]["text"],
        "n_raters": n_raters,
        "top_vote_count": max_count,
        "top_labels": " | ".join(top_labels),
        "exclude_due_to_ne_mogu_da_razumem_top": str(exclude_item).lower(),
        "strict_majority_label": strict_majority,
        "unique_highest_frequency_label": unique_plurality,
        "chatgpt_label": chatgpt_label,
        "matches_majority": str(bool(strict_majority) and strict_majority == chatgpt_label).lower(),
        "matches_unique_highest_frequency": str(bool(unique_plurality) and unique_plurality == chatgpt_label).lower(),
        "label_counts_json": json.dumps(
            {label: counts[label] for label in CATEGORIES if counts[label]},
            ensure_ascii=False,
        ),
    }


def group_items(
    rows: list[dict[str, str]]
) -> dict[int, dict[str, list[dict[str, str]]]]:
    grouped: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["sample_idx"])][row["tweet_id"]].append(row)
    return grouped


def fleiss_kappa(item_counts: list[Counter[str]]) -> dict[str, float]:
    if not item_counts:
        return {"observed_agreement": math.nan, "expected_agreement": math.nan, "kappa": math.nan}
    n_raters = sum(item_counts[0].values())
    n_items = len(item_counts)
    p_bar = 0.0
    category_totals = Counter()
    for counts in item_counts:
        sum_sq = sum(counts.get(label, 0) ** 2 for label in CATEGORIES)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        p_bar += p_i
        category_totals.update(counts)
    p_bar /= n_items
    total = n_items * n_raters
    p_e = sum((category_totals.get(label, 0) / total) ** 2 for label in CATEGORIES)
    kappa = (p_bar - p_e) / (1 - p_e) if (1 - p_e) else math.nan
    return {"observed_agreement": p_bar, "expected_agreement": p_e, "kappa": kappa}


def krippendorff_alpha_nominal(item_counts: list[Counter[str]]) -> dict[str, float]:
    if not item_counts:
        return {"observed_disagreement": math.nan, "expected_disagreement": math.nan, "alpha": math.nan}
    pooled = Counter()
    do_num = 0.0
    do_den = 0.0
    for counts in item_counts:
        n_i = sum(counts.values())
        same = sum(v * (v - 1) for v in counts.values())
        do_num += n_i * (n_i - 1) - same
        do_den += n_i * (n_i - 1)
        pooled.update(counts)
    total = sum(pooled.values())
    same_total = sum(v * (v - 1) for v in pooled.values())
    observed = do_num / do_den
    expected = (total * (total - 1) - same_total) / (total * (total - 1))
    alpha = 1 - observed / expected if expected else math.nan
    return {"observed_disagreement": observed, "expected_disagreement": expected, "alpha": alpha}


def cohen_kappa(xs: list[str], ys: list[str], labels: list[str]) -> tuple[float, float]:
    n = len(xs)
    observed = sum(a == b for a, b in zip(xs, ys)) / n
    counts_x = Counter(xs)
    counts_y = Counter(ys)
    expected = sum((counts_x.get(label, 0) / n) * (counts_y.get(label, 0) / n) for label in labels)
    kappa = (observed - expected) / (1 - expected) if (1 - expected) else math.nan
    return observed, kappa


def filtered_sample_summary(item_rows_by_tweet: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    item_counts = [label_counts(rows) for _, rows in sorted(item_rows_by_tweet.items())]
    coder_map: dict[str, dict[str, str]] = defaultdict(dict)
    coders = set()
    tweets = sorted(item_rows_by_tweet)
    for tweet_id, rows in item_rows_by_tweet.items():
        for row in rows:
            coder_map[row["annotator_code"]][tweet_id] = row["label"]
            coders.add(row["annotator_code"])
    coders = sorted(coders)
    pairwise_agreements = []
    pairwise_kappas = []
    for coder_a, coder_b in itertools.combinations(coders, 2):
        labels_a = [coder_map[coder_a][tweet_id] for tweet_id in tweets]
        labels_b = [coder_map[coder_b][tweet_id] for tweet_id in tweets]
        agreement, kappa = cohen_kappa(labels_a, labels_b, CATEGORIES)
        pairwise_agreements.append(agreement)
        pairwise_kappas.append(kappa)

    fleiss = fleiss_kappa(item_counts)
    alpha = krippendorff_alpha_nominal(item_counts)
    return {
        "n_items": len(item_rows_by_tweet),
        "n_raters": len(coders),
        "pair_count": math.comb(len(coders), 2),
        "mean_pairwise_agreement": sum(pairwise_agreements) / len(pairwise_agreements),
        "mean_pairwise_kappa": sum(pairwise_kappas) / len(pairwise_kappas),
        "fleiss_kappa": fleiss["kappa"],
        "fleiss_observed_agreement": fleiss["observed_agreement"],
        "krippendorff_alpha_nominal": alpha["alpha"],
    }


def consensus_vs_chatgpt(records: list[dict[str, object]], label_key: str) -> dict[str, object]:
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
    top_errors = Counter((truth, pred) for truth, pred in zip(y_true, y_pred) if truth != pred)
    return {
        "n_items": len(rows),
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "macro_recall": macro_recall,
        "top_errors": [
            {"consensus_label": truth, "chatgpt_label": pred, "count": count}
            for (truth, pred), count in top_errors.most_common(15)
        ],
    }


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    annotations = read_annotations()
    chatgpt = read_chatgpt()
    grouped = group_items(annotations)

    item_records = []
    filtered_by_sample: dict[int, dict[str, list[dict[str, str]]]] = {}
    removed_counts = {}

    for sample_idx in SAMPLES:
        sample_items = grouped[sample_idx]
        filtered_items = {}
        removed = 0
        for tweet_id, rows in sample_items.items():
            record = item_record(rows, chatgpt)
            item_records.append(record)
            if record["exclude_due_to_ne_mogu_da_razumem_top"] == "true":
                removed += 1
            else:
                filtered_items[tweet_id] = rows
        filtered_by_sample[sample_idx] = filtered_items
        removed_counts[sample_idx] = removed

    item_records.sort(key=lambda record: (record["sample_idx"], record["tweet_pos"]))
    with ITEMS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
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
                "matches_majority",
                "matches_unique_highest_frequency",
                "label_counts_json",
            ],
        )
        writer.writeheader()
        for record in item_records:
            writer.writerow(record)

    filtered_sample_metrics = {
        sample_idx: filtered_sample_summary(filtered_by_sample[sample_idx])
        for sample_idx in SAMPLES
    }

    weighted_den = sum(
        metrics["pair_count"] * metrics["n_items"]
        for metrics in filtered_sample_metrics.values()
    )
    all_filtered_item_counts = []
    for sample_idx in SAMPLES:
        all_filtered_item_counts.extend(
            label_counts(rows)
            for _, rows in sorted(filtered_by_sample[sample_idx].items())
        )
    overall_alpha = krippendorff_alpha_nominal(all_filtered_item_counts)["alpha"]
    overall_agreement = sum(
        metrics["mean_pairwise_agreement"] * metrics["pair_count"] * metrics["n_items"]
        for metrics in filtered_sample_metrics.values()
    ) / weighted_den
    overall_pairwise_kappa = sum(
        metrics["mean_pairwise_kappa"] * metrics["pair_count"] * metrics["n_items"]
        for metrics in filtered_sample_metrics.values()
    ) / weighted_den

    filtered_records = [
        record
        for record in item_records
        if record["exclude_due_to_ne_mogu_da_razumem_top"] == "false"
    ]
    majority_metrics = consensus_vs_chatgpt(filtered_records, "strict_majority_label")
    plurality_metrics = consensus_vs_chatgpt(filtered_records, "unique_highest_frequency_label")

    per_sample_consensus = {}
    for sample_idx in SAMPLES:
        sample_records = [
            record
            for record in filtered_records
            if record["sample_idx"] == sample_idx
        ]
        per_sample_consensus[sample_idx] = {
            "majority": consensus_vs_chatgpt(sample_records, "strict_majority_label"),
            "unique_highest_frequency": consensus_vs_chatgpt(
                sample_records, "unique_highest_frequency_label"
            ),
        }

    metrics_payload = {
        "removed_counts": removed_counts,
        "filtered_sample_metrics": filtered_sample_metrics,
        "filtered_overall": {
            "n_items": len(filtered_records),
            "weighted_mean_pairwise_agreement": overall_agreement,
            "weighted_mean_pairwise_kappa": overall_pairwise_kappa,
            "overall_krippendorff_alpha_nominal": overall_alpha,
        },
        "chatgpt_comparison": {
            "majority": majority_metrics,
            "unique_highest_frequency": plurality_metrics,
            "by_sample": per_sample_consensus,
        },
    }

    with METRICS_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Samples 18-25 Consensus Analysis",
        "",
        "Assumption: items are excluded if `Ne mogu da razumem` is tied for the highest vote count or is the unique highest vote count.",
        "",
        "## Filtered Agreement",
        "",
    ]
    for sample_idx in SAMPLES:
        metrics = filtered_sample_metrics[sample_idx]
        kept = metrics["n_items"]
        removed = removed_counts[sample_idx]
        lines.extend(
            [
                f"### Sample {sample_idx}",
                "",
                f"- Kept items: {kept}",
                f"- Removed items: {removed}",
                f"- Mean pairwise agreement: {fmt(metrics['mean_pairwise_agreement'])}",
                f"- Mean pairwise Cohen's kappa: {fmt(metrics['mean_pairwise_kappa'])}",
                f"- Fleiss' kappa: {fmt(metrics['fleiss_kappa'])}",
                f"- Krippendorff's alpha (nominal): {fmt(metrics['krippendorff_alpha_nominal'])}",
                "",
            ]
        )

    lines.extend(
        [
            "### Overall",
            "",
            f"- Kept items total: {len(filtered_records)}",
            f"- Removed items total: {sum(removed_counts.values())}",
            f"- Weighted mean pairwise agreement: {fmt(overall_agreement)}",
            f"- Weighted mean pairwise Cohen's kappa: {fmt(overall_pairwise_kappa)}",
            f"- Overall Krippendorff's alpha (nominal): {fmt(overall_alpha)}",
            "",
            "## ChatGPT Comparison",
            "",
            f"- Majority subset items: {majority_metrics['n_items']}",
            f"- Majority accuracy: {fmt(majority_metrics['accuracy'])}",
            f"- Majority Cohen's kappa: {fmt(majority_metrics['cohen_kappa'])}",
            f"- Majority macro recall: {fmt(majority_metrics['macro_recall'])}",
            "",
            f"- Unique-highest-frequency subset items: {plurality_metrics['n_items']}",
            f"- Unique-highest-frequency accuracy: {fmt(plurality_metrics['accuracy'])}",
            f"- Unique-highest-frequency Cohen's kappa: {fmt(plurality_metrics['cohen_kappa'])}",
            f"- Unique-highest-frequency macro recall: {fmt(plurality_metrics['macro_recall'])}",
            "",
            "## ChatGPT by Sample",
            "",
        ]
    )

    for sample_idx in SAMPLES:
        majority = per_sample_consensus[sample_idx]["majority"]
        plurality = per_sample_consensus[sample_idx]["unique_highest_frequency"]
        lines.extend(
            [
                f"### Sample {sample_idx}",
                "",
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
