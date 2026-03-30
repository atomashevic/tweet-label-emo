import json
import random

import pandas as pd
import streamlit as st

from config import LABEL_NORM

SEED = 42
NUM_SAMPLES = 10

SCL_SEED = 99
SCL_NUM_SAMPLES = 3
SCL_SAMPLE_SIZE = 50

FF_SEED = 123
FF_NUM_SAMPLES = 5
FF_SAMPLE_SIZE = 50

FF2_SEED = 177
FF2_NUM_SAMPLES = 11
FF2_SAMPLE_SIZE = 50


@st.cache_data
def load_tweets() -> pd.DataFrame:
    return pd.read_csv("data.csv", dtype=str).reset_index(drop=True)


@st.cache_data
def load_labeled_tweets() -> pd.DataFrame:
    """Returns merged DataFrame with columns: tweet_id, text, chatgpt_emotion"""
    tweets = load_tweets().set_index("tweet_id")
    labels = []
    with open("emotions_chatgpt.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            labels.append({
                "tweet_id": obj["id_str"],
                "chatgpt_emotion": LABEL_NORM.get(obj["emotion"], obj["emotion"]),
            })
    labels_df = pd.DataFrame(labels).set_index("tweet_id")
    merged = tweets.join(labels_df, how="inner").reset_index()
    return merged


@st.cache_data
def get_samples() -> list[pd.DataFrame]:
    """Returns list of 10 DataFrames, one per annotator sample."""
    df = load_labeled_tweets()
    rng = random.Random(SEED)

    # Group by emotion, shuffle each group
    groups: dict[str, list[int]] = {}
    for emotion, group in df.groupby("chatgpt_emotion"):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        groups[emotion] = idxs

    # Distribute: for each emotion, slice into NUM_SAMPLES chunks
    sample_indices: list[list[int]] = [[] for _ in range(NUM_SAMPLES)]
    for emotion, idxs in groups.items():
        chunk = len(idxs) // NUM_SAMPLES
        for i in range(NUM_SAMPLES):
            start = i * chunk
            end = start + chunk if i < NUM_SAMPLES - 1 else len(idxs)
            sample_indices[i].extend(idxs[start:end])

    # Shuffle each sample for natural presentation order
    for i in range(NUM_SAMPLES):
        rng.shuffle(sample_indices[i])

    return [df.iloc[sample_indices[i]].reset_index(drop=True) for i in range(NUM_SAMPLES)]


@st.cache_data
def get_scl_samples() -> list[pd.DataFrame]:
    """Returns 3 stratified 50-tweet samples for scl annotators (indices 10-12)."""
    df = load_labeled_tweets()
    rng = random.Random(SCL_SEED)
    total = len(df)

    sample_indices: list[list[int]] = [[] for _ in range(SCL_NUM_SAMPLES)]
    for emotion, group in df.groupby("chatgpt_emotion"):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        per_sample = max(1, round(SCL_SAMPLE_SIZE * len(idxs) / total))
        needed = per_sample * SCL_NUM_SAMPLES
        idxs = idxs[:needed]
        for i in range(SCL_NUM_SAMPLES):
            sample_indices[i].extend(idxs[i * per_sample : (i + 1) * per_sample])

    for i in range(SCL_NUM_SAMPLES):
        rng.shuffle(sample_indices[i])

    return [df.iloc[sample_indices[i]].reset_index(drop=True) for i in range(SCL_NUM_SAMPLES)]


@st.cache_data
def get_ff_samples() -> list[pd.DataFrame]:
    """Returns 5 stratified 50-tweet samples for ff annotators (indices 13-17)."""
    df = load_labeled_tweets()
    rng = random.Random(FF_SEED)
    total = len(df)

    sample_indices: list[list[int]] = [[] for _ in range(FF_NUM_SAMPLES)]
    for emotion, group in df.groupby("chatgpt_emotion"):
        idxs = group.index.tolist()
        rng.shuffle(idxs)
        per_sample = max(1, round(FF_SAMPLE_SIZE * len(idxs) / total))
        needed = per_sample * FF_NUM_SAMPLES
        idxs = idxs[:needed]
        for i in range(FF_NUM_SAMPLES):
            sample_indices[i].extend(idxs[i * per_sample : (i + 1) * per_sample])

    for i in range(FF_NUM_SAMPLES):
        rng.shuffle(sample_indices[i])

    return [df.iloc[sample_indices[i]].reset_index(drop=True) for i in range(FF_NUM_SAMPLES)]


def _fixed_sample_quota(
    df: pd.DataFrame, sample_size: int
) -> dict[str, int]:
    """Return a stable per-emotion quota that sums exactly to sample_size."""
    counts = df["chatgpt_emotion"].value_counts().sort_index()
    total = len(df)

    quotas = {
        emotion: int(sample_size * count / total)
        for emotion, count in counts.items()
    }
    assigned = sum(quotas.values())

    remainders = sorted(
        (
            (sample_size * count / total) - quotas[emotion],
            emotion,
        )
        for emotion, count in counts.items()
    )
    extras = sample_size - assigned
    if extras > 0:
        for _, emotion in reversed(remainders[-extras:]):
            quotas[emotion] += 1

    return quotas


@st.cache_data
def get_ff2_samples() -> list[pd.DataFrame]:
    """Returns 11 non-overlapping 50-tweet samples for ff101-ff177."""
    df = load_labeled_tweets()
    rng = random.Random(FF2_SEED)
    quotas = _fixed_sample_quota(df, FF2_SAMPLE_SIZE)

    sample_indices: list[list[int]] = [[] for _ in range(FF2_NUM_SAMPLES)]
    for emotion, quota in quotas.items():
        idxs = df.index[df["chatgpt_emotion"] == emotion].tolist()
        rng.shuffle(idxs)
        idxs = idxs[: quota * FF2_NUM_SAMPLES]
        for i in range(FF2_NUM_SAMPLES):
            start = i * quota
            end = start + quota
            sample_indices[i].extend(idxs[start:end])

    for i in range(FF2_NUM_SAMPLES):
        rng.shuffle(sample_indices[i])

    return [df.iloc[sample_indices[i]].reset_index(drop=True) for i in range(FF2_NUM_SAMPLES)]


def get_sample(sample_idx: int) -> "pd.DataFrame":
    """Return the correct sample DataFrame for any supported sample index."""
    if sample_idx < NUM_SAMPLES:
        return get_samples()[sample_idx]
    if sample_idx < NUM_SAMPLES + SCL_NUM_SAMPLES:
        return get_scl_samples()[sample_idx - NUM_SAMPLES]
    ff_start = NUM_SAMPLES + SCL_NUM_SAMPLES
    ff_end = ff_start + FF_NUM_SAMPLES
    if sample_idx < ff_end:
        return get_ff_samples()[sample_idx - ff_start]
    return get_ff2_samples()[sample_idx - ff_end]
