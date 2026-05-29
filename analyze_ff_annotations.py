import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from config import EMOTIONS, SPECIAL_LABELS


INPUT_PATH = Path("ff101_ff125_annotations.csv")
MAJORITY_OUTPUT_PATH = Path("ff101_ff125_majority_vote.csv")
REPORT_OUTPUT_PATH = Path("ff101_ff125_agreement_report.md")
JSON_OUTPUT_PATH = Path("ff101_ff125_agreement_metrics.json")

CATEGORIES = EMOTIONS + SPECIAL_LABELS


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_rows(
    rows: list[dict[str, str]],
) -> tuple[dict[int, list[dict[str, str]]], dict[tuple[int, str], list[dict[str, str]]]]:
    by_sample: dict[int, list[dict[str, str]]] = defaultdict(list)
    by_item: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sample_idx = int(row["sample_idx"])
        key = (sample_idx, row["tweet_id"])
        by_sample[sample_idx].append(row)
        by_item[key].append(row)
    return by_sample, by_item


def label_counter(rows: list[dict[str, str]]) -> Counter[str]:
    counts = Counter(row["label"] for row in rows)
    unknown = sorted(label for label in counts if label not in CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown labels encountered: {unknown}")
    return counts


def majority_record(
    item_rows: list[dict[str, str]],
) -> dict[str, object]:
    counts = label_counter(item_rows)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top_count = ordered[0][1]
    top_labels = [label for label, count in ordered if count == top_count]
    tweet_pos = min(int(row["tweet_pos"]) for row in item_rows)

    return {
        "sample_idx": int(item_rows[0]["sample_idx"]),
        "tweet_pos": tweet_pos,
        "tweet_id": item_rows[0]["tweet_id"],
        "text": item_rows[0]["text"],
        "n_raters": len(item_rows),
        "majority_label": " | ".join(top_labels),
        "majority_vote_count": top_count,
        "majority_vote_share": f"{top_count / len(item_rows):.4f}",
        "is_tie": str(len(top_labels) > 1).lower(),
        "tied_labels": " | ".join(top_labels) if len(top_labels) > 1 else "",
        "label_counts_json": json.dumps(
            {label: counts[label] for label in CATEGORIES if counts[label]},
            ensure_ascii=False,
            sort_keys=False,
        ),
    }


def fleiss_kappa(item_counts: list[Counter[str]]) -> dict[str, float]:
    if not item_counts:
        return {"observed_agreement": math.nan, "expected_agreement": math.nan, "kappa": math.nan}

    n_raters = sum(item_counts[0].values())
    if any(sum(counts.values()) != n_raters for counts in item_counts):
        raise ValueError("Fleiss' kappa requires a fixed number of raters per item.")

    n_items = len(item_counts)
    p_bar_sum = 0.0
    category_totals = Counter()

    for counts in item_counts:
        sum_sq = sum(counts[label] ** 2 for label in CATEGORIES)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        p_bar_sum += p_i
        category_totals.update(counts)

    p_bar = p_bar_sum / n_items
    total_ratings = n_items * n_raters
    p_e = sum((category_totals[label] / total_ratings) ** 2 for label in CATEGORIES)
    denom = 1.0 - p_e
    kappa = (p_bar - p_e) / denom if denom else math.nan
    return {
        "observed_agreement": p_bar,
        "expected_agreement": p_e,
        "kappa": kappa,
    }


def krippendorff_alpha_nominal(item_counts: list[Counter[str]]) -> dict[str, float]:
    if not item_counts:
        return {"observed_disagreement": math.nan, "expected_disagreement": math.nan, "alpha": math.nan}

    observed_num = 0.0
    observed_den = 0.0
    pooled = Counter()

    for counts in item_counts:
        n_i = sum(counts.values())
        if n_i < 2:
            continue
        same_label_pairs = sum(count * (count - 1) for count in counts.values())
        observed_num += n_i * (n_i - 1) - same_label_pairs
        observed_den += n_i * (n_i - 1)
        pooled.update(counts)

    total = sum(pooled.values())
    if observed_den == 0 or total < 2:
        return {"observed_disagreement": math.nan, "expected_disagreement": math.nan, "alpha": math.nan}

    same_expected = sum(count * (count - 1) for count in pooled.values())
    observed_disagreement = observed_num / observed_den
    expected_disagreement = (total * (total - 1) - same_expected) / (total * (total - 1))
    alpha = 1.0 - (observed_disagreement / expected_disagreement) if expected_disagreement else math.nan
    return {
        "observed_disagreement": observed_disagreement,
        "expected_disagreement": expected_disagreement,
        "alpha": alpha,
    }


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> tuple[float, float]:
    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's kappa requires aligned rating vectors.")
    if not labels_a:
        return math.nan, math.nan

    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / total) * (counts_b[label] / total) for label in CATEGORIES)
    denom = 1.0 - expected
    kappa = (observed - expected) / denom if denom else math.nan
    return observed, kappa


def sample_metrics(sample_rows: list[dict[str, str]]) -> dict[str, object]:
    coders = sorted({row["annotator_code"] for row in sample_rows})
    items = sorted({row["tweet_id"] for row in sample_rows})

    coder_labels: dict[str, dict[str, str]] = {
        coder: {} for coder in coders
    }
    item_counts: list[Counter[str]] = []
    unanimous = 0
    strict_majority = 0
    ties = 0

    rows_by_tweet: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        rows_by_tweet[row["tweet_id"]].append(row)
        coder_labels[row["annotator_code"]][row["tweet_id"]] = row["label"]

    for tweet_id in items:
        counts = label_counter(rows_by_tweet[tweet_id])
        item_counts.append(counts)
        ordered = sorted(counts.values(), reverse=True)
        if ordered[0] == len(rows_by_tweet[tweet_id]):
            unanimous += 1
        if ordered[0] > len(rows_by_tweet[tweet_id]) / 2:
            strict_majority += 1
        if len(ordered) > 1 and ordered[0] == ordered[1]:
            ties += 1

    pairwise_agreements = []
    pairwise_kappas = []
    for coder_a, coder_b in itertools.combinations(coders, 2):
        labels_a = [coder_labels[coder_a][tweet_id] for tweet_id in items]
        labels_b = [coder_labels[coder_b][tweet_id] for tweet_id in items]
        agreement, kappa = cohen_kappa(labels_a, labels_b)
        pairwise_agreements.append(agreement)
        pairwise_kappas.append(kappa)

    fleiss = fleiss_kappa(item_counts)
    alpha = krippendorff_alpha_nominal(item_counts)

    return {
        "n_items": len(items),
        "n_raters": len(coders),
        "coders": coders,
        "unanimous_items": unanimous,
        "strict_majority_items": strict_majority,
        "tie_items": ties,
        "mean_pairwise_agreement": sum(pairwise_agreements) / len(pairwise_agreements),
        "mean_pairwise_cohen_kappa": sum(pairwise_kappas) / len(pairwise_kappas),
        "fleiss_kappa": fleiss["kappa"],
        "fleiss_observed_agreement": fleiss["observed_agreement"],
        "krippendorff_alpha_nominal": alpha["alpha"],
        "krippendorff_observed_disagreement": alpha["observed_disagreement"],
    }


def overall_metrics(by_sample: dict[int, list[dict[str, str]]]) -> dict[str, object]:
    all_item_counts: list[Counter[str]] = []
    weighted_pairwise_agreement_num = 0.0
    weighted_pairwise_kappa_num = 0.0
    weighted_pairwise_den = 0
    total_items = 0
    total_unanimous = 0
    total_majority = 0
    total_ties = 0

    per_sample = {}
    for sample_idx, sample_rows in sorted(by_sample.items()):
        metrics = sample_metrics(sample_rows)
        per_sample[sample_idx] = metrics
        total_items += metrics["n_items"]
        total_unanimous += metrics["unanimous_items"]
        total_majority += metrics["strict_majority_items"]
        total_ties += metrics["tie_items"]

        rows_by_tweet: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in sample_rows:
            rows_by_tweet[row["tweet_id"]].append(row)
        for tweet_id in sorted(rows_by_tweet):
            all_item_counts.append(label_counter(rows_by_tweet[tweet_id]))

        pair_count = math.comb(metrics["n_raters"], 2)
        weighted_pairwise_agreement_num += metrics["mean_pairwise_agreement"] * pair_count * metrics["n_items"]
        weighted_pairwise_kappa_num += metrics["mean_pairwise_cohen_kappa"] * pair_count * metrics["n_items"]
        weighted_pairwise_den += pair_count * metrics["n_items"]

    alpha = krippendorff_alpha_nominal(all_item_counts)
    return {
        "samples": per_sample,
        "total_items": total_items,
        "total_unanimous_items": total_unanimous,
        "total_strict_majority_items": total_majority,
        "total_tie_items": total_ties,
        "weighted_mean_pairwise_agreement": weighted_pairwise_agreement_num / weighted_pairwise_den,
        "weighted_mean_pairwise_cohen_kappa": weighted_pairwise_kappa_num / weighted_pairwise_den,
        "overall_krippendorff_alpha_nominal": alpha["alpha"],
        "overall_krippendorff_observed_disagreement": alpha["observed_disagreement"],
    }


def write_majority_vote_csv(
    by_item: dict[tuple[int, str], list[dict[str, str]]],
) -> list[dict[str, object]]:
    records = [majority_record(rows) for _, rows in sorted(by_item.items())]
    records.sort(key=lambda row: (row["sample_idx"], row["tweet_pos"]))
    with MAJORITY_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "tweet_pos",
                "tweet_id",
                "text",
                "n_raters",
                "majority_label",
                "majority_vote_count",
                "majority_vote_share",
                "is_tie",
                "tied_labels",
                "label_counts_json",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return records


def write_metrics_json(metrics: dict[str, object]) -> None:
    with JSON_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def write_report(metrics: dict[str, object], majority_records: list[dict[str, object]]) -> None:
    majority_counts = Counter(record["majority_label"] for record in majority_records)
    tie_count = sum(record["is_tie"] == "true" for record in majority_records)

    lines = [
        "# FF101-FF125 Annotation Agreement",
        "",
        "## Overall",
        "",
        f"- Total tweet items: {metrics['total_items']}",
        f"- Unique-majority items: {metrics['total_strict_majority_items']}",
        f"- Tied-top-vote items: {metrics['total_tie_items']}",
        f"- Unanimous items: {metrics['total_unanimous_items']}",
        f"- Weighted mean pairwise agreement: {fmt(metrics['weighted_mean_pairwise_agreement'])}",
        f"- Weighted mean pairwise Cohen's kappa: {fmt(metrics['weighted_mean_pairwise_cohen_kappa'])}",
        f"- Overall Krippendorff's alpha (nominal): {fmt(metrics['overall_krippendorff_alpha_nominal'])}",
        "",
        "## By Sample",
        "",
    ]

    for sample_idx, sample_metrics_dict in metrics["samples"].items():
        lines.extend(
            [
                f"### Sample {sample_idx}",
                "",
                f"- Raters: {sample_metrics_dict['n_raters']}",
                f"- Items: {sample_metrics_dict['n_items']}",
                f"- Unanimous items: {sample_metrics_dict['unanimous_items']}",
                f"- Unique-majority items: {sample_metrics_dict['strict_majority_items']}",
                f"- Tied-top-vote items: {sample_metrics_dict['tie_items']}",
                f"- Mean pairwise agreement: {fmt(sample_metrics_dict['mean_pairwise_agreement'])}",
                f"- Mean pairwise Cohen's kappa: {fmt(sample_metrics_dict['mean_pairwise_cohen_kappa'])}",
                f"- Fleiss' kappa: {fmt(sample_metrics_dict['fleiss_kappa'])}",
                f"- Fleiss observed agreement: {fmt(sample_metrics_dict['fleiss_observed_agreement'])}",
                f"- Krippendorff's alpha (nominal): {fmt(sample_metrics_dict['krippendorff_alpha_nominal'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Majority Vote Distribution",
            "",
            f"- Items with any tie at the top vote count: {tie_count}",
        ]
    )
    for label, count in sorted(majority_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {label}: {count}")

    REPORT_OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = read_rows(INPUT_PATH)
    by_sample, by_item = group_rows(rows)
    majority_records = write_majority_vote_csv(by_item)
    metrics = overall_metrics(by_sample)
    write_metrics_json(metrics)
    write_report(metrics, majority_records)

    print(f"Wrote {MAJORITY_OUTPUT_PATH}")
    print(f"Wrote {REPORT_OUTPUT_PATH}")
    print(f"Wrote {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
