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


def get_sample(sample_idx: int) -> "pd.DataFrame":
    """Return the correct sample DataFrame for any sample index (0-9 main, 10-12 scl, 13-17 ff)."""
    if sample_idx < NUM_SAMPLES:
        return get_samples()[sample_idx]
    if sample_idx < NUM_SAMPLES + SCL_NUM_SAMPLES:
        return get_scl_samples()[sample_idx - NUM_SAMPLES]
    return get_ff_samples()[sample_idx - NUM_SAMPLES - SCL_NUM_SAMPLES]
