import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ANNOTATIONS_PATH = Path("ff_samples_18_22_annotations.csv")
REPORT_PATH = Path("samples_18_22_ignore_dontknow_report.md")
METRICS_PATH = Path("samples_18_22_ignore_dontknow_metrics.json")

SAMPLES = [18, 19, 20, 21, 22]
DROP_LABEL = "Ne mogu da razumem"
CATEGORIES = [
    "poverenje",
    "bes",
    "tuga",
    "iznenađenje",
    "strah",
    "gađenje",
    "radost",
    "iščekivanje",
    "Emocionalno neutralno",
]


def read_annotations() -> list[dict[str, str]]:
    with ANNOTATIONS_PATH.open(newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if int(row["sample_idx"]) in SAMPLES]


def group_by_sample_item(
    rows: list[dict[str, str]],
) -> dict[int, dict[str, list[dict[str, str]]]]:
    grouped: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[int(row["sample_idx"])][row["tweet_id"]].append(row)
    return grouped


def alpha_nominal_variable(item_counts: list[Counter[str]]) -> dict[str, float]:
    pooled = Counter()
    do_num = 0.0
    do_den = 0.0
    usable_items = 0
    for counts in item_counts:
        n_i = sum(counts.values())
        if n_i < 2:
            continue
        usable_items += 1
        same = sum(v * (v - 1) for v in counts.values())
        do_num += n_i * (n_i - 1) - same
        do_den += n_i * (n_i - 1)
        pooled.update(counts)

    total = sum(pooled.values())
    if do_den == 0 or total < 2:
        return {
            "usable_items": usable_items,
            "observed_disagreement": math.nan,
            "expected_disagreement": math.nan,
            "alpha": math.nan,
        }

    same_total = sum(v * (v - 1) for v in pooled.values())
    observed = do_num / do_den
    expected = (total * (total - 1) - same_total) / (total * (total - 1))
    alpha = 1 - observed / expected if expected else math.nan
    return {
        "usable_items": usable_items,
        "observed_disagreement": observed,
        "expected_disagreement": expected,
        "alpha": alpha,
    }


def pairwise_overlap_metrics(
    sample_items: dict[str, list[dict[str, str]]]
) -> dict[str, float]:
    coder_map: dict[str, dict[str, str]] = defaultdict(dict)
    coders = set()
    for tweet_id, rows in sample_items.items():
        for row in rows:
            if row["label"] == DROP_LABEL:
                continue
            coder_map[row["annotator_code"]][tweet_id] = row["label"]
            coders.add(row["annotator_code"])

    coders = sorted(coders)
    agreements = []
    kappas = []
    overlaps = []
    for coder_a, coder_b in itertools.combinations(coders, 2):
        common = sorted(set(coder_map[coder_a]) & set(coder_map[coder_b]))
        if not common:
            continue
        labels_a = [coder_map[coder_a][tweet_id] for tweet_id in common]
        labels_b = [coder_map[coder_b][tweet_id] for tweet_id in common]
        n = len(common)
        observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
        counts_a = Counter(labels_a)
        counts_b = Counter(labels_b)
        expected = sum(
            (counts_a.get(label, 0) / n) * (counts_b.get(label, 0) / n)
            for label in CATEGORIES
        )
        kappa = (observed - expected) / (1 - expected) if (1 - expected) else math.nan
        agreements.append(observed)
        kappas.append(kappa)
        overlaps.append(n)

    total_overlap = sum(overlaps)
    weighted_agreement = (
        sum(agreement * overlap for agreement, overlap in zip(agreements, overlaps)) / total_overlap
        if total_overlap
        else math.nan
    )
    weighted_kappa = (
        sum(kappa * overlap for kappa, overlap in zip(kappas, overlaps)) / total_overlap
        if total_overlap
        else math.nan
    )
    return {
        "n_coders": len(coders),
        "n_pairs": len(overlaps),
        "total_pairwise_overlap": total_overlap,
        "weighted_pairwise_agreement": weighted_agreement,
        "weighted_pairwise_kappa": weighted_kappa,
    }


def summarize_sample(sample_items: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    item_counts: list[Counter[str]] = []
    dropped_votes = 0
    items_with_any_drop = 0
    items_with_lt2 = 0

    for _, rows in sorted(sample_items.items()):
        counts = Counter()
        had_drop = False
        for row in rows:
            if row["label"] == DROP_LABEL:
                dropped_votes += 1
                had_drop = True
                continue
            counts[row["label"]] += 1
        if had_drop:
            items_with_any_drop += 1
        if sum(counts.values()) < 2:
            items_with_lt2 += 1
        item_counts.append(counts)

    alpha = alpha_nominal_variable(item_counts)
    pairwise = pairwise_overlap_metrics(sample_items)
    return {
        "items_total": len(sample_items),
        "dropped_dontknow_votes": dropped_votes,
        "items_with_any_dontknow_vote": items_with_any_drop,
        "items_with_lt2_remaining_votes": items_with_lt2,
        "usable_items_for_alpha": alpha["usable_items"],
        "weighted_pairwise_agreement_overlap": pairwise["weighted_pairwise_agreement"],
        "weighted_pairwise_kappa_overlap": pairwise["weighted_pairwise_kappa"],
        "krippendorff_alpha_nominal_no_dontknow": alpha["alpha"],
        "total_pairwise_overlap": pairwise["total_pairwise_overlap"],
        "item_counts": item_counts,
    }


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def main() -> None:
    rows = read_annotations()
    grouped = group_by_sample_item(rows)
    sample_metrics = {sample: summarize_sample(grouped[sample]) for sample in SAMPLES}

    overall_item_counts = []
    weighted_agreement_num = 0.0
    weighted_kappa_num = 0.0
    weighted_den = 0
    for sample in SAMPLES:
        metrics = sample_metrics[sample]
        overall_item_counts.extend(metrics["item_counts"])
        weighted_agreement_num += (
            metrics["weighted_pairwise_agreement_overlap"] * metrics["total_pairwise_overlap"]
        )
        weighted_kappa_num += (
            metrics["weighted_pairwise_kappa_overlap"] * metrics["total_pairwise_overlap"]
        )
        weighted_den += metrics["total_pairwise_overlap"]

    overall_alpha = alpha_nominal_variable(overall_item_counts)
    overall = {
        "usable_items_for_alpha": overall_alpha["usable_items"],
        "weighted_pairwise_agreement_overlap": weighted_agreement_num / weighted_den,
        "weighted_pairwise_kappa_overlap": weighted_kappa_num / weighted_den,
        "krippendorff_alpha_nominal_no_dontknow": overall_alpha["alpha"],
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "Drop only `Ne mogu da razumem` votes and treat them as missing. Agreement is computed over emotions + neutral only.",
                "samples": {
                    sample: {k: v for k, v in metrics.items() if k != "item_counts"}
                    for sample, metrics in sample_metrics.items()
                },
                "overall": overall,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "# Samples 18-22 Agreement Ignoring `Ne mogu da razumem`",
        "",
        "Method: drop only the `Ne mogu da razumem` votes and treat them as missing data. Agreement is computed on the remaining emotion + neutral labels.",
        "",
    ]
    for sample in SAMPLES:
        metrics = sample_metrics[sample]
        lines.extend(
            [
                f"## Sample {sample}",
                "",
                f"- Items total: {metrics['items_total']}",
                f"- Dropped `Ne mogu da razumem` votes: {metrics['dropped_dontknow_votes']}",
                f"- Items with any dropped vote: {metrics['items_with_any_dontknow_vote']}",
                f"- Items with fewer than 2 remaining votes: {metrics['items_with_lt2_remaining_votes']}",
                f"- Usable items for alpha: {metrics['usable_items_for_alpha']}",
                f"- Weighted pairwise agreement on overlap: {fmt(metrics['weighted_pairwise_agreement_overlap'])}",
                f"- Weighted pairwise Cohen's kappa on overlap: {fmt(metrics['weighted_pairwise_kappa_overlap'])}",
                f"- Krippendorff's alpha (nominal, no `Ne mogu da razumem`): {fmt(metrics['krippendorff_alpha_nominal_no_dontknow'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Overall",
            "",
            f"- Usable items for alpha: {overall['usable_items_for_alpha']}",
            f"- Weighted pairwise agreement on overlap: {fmt(overall['weighted_pairwise_agreement_overlap'])}",
            f"- Weighted pairwise Cohen's kappa on overlap: {fmt(overall['weighted_pairwise_kappa_overlap'])}",
            f"- Krippendorff's alpha (nominal, no `Ne mogu da razumem`): {fmt(overall['krippendorff_alpha_nominal_no_dontknow'])}",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
