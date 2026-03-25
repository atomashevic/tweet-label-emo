import streamlit as st
from supabase import create_client


@st.cache_resource
def get_client():
    return create_client(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"],
    )


def save_annotation(
    code: str,
    sample_idx: int,
    tweet_pos: int,
    tweet_id: str,
    label: str,
) -> None:
    get_client().table("annotations").upsert(
        {
            "annotator_code": code,
            "sample_idx": sample_idx,
            "tweet_pos": tweet_pos,
            "tweet_id": tweet_id,
            "label": label,
        },
        on_conflict="annotator_code,tweet_pos",
    ).execute()


def load_annotations(code: str) -> dict[int, str]:
    res = (
        get_client()
        .table("annotations")
        .select("tweet_pos,label")
        .eq("annotator_code", code)
        .execute()
    )
    return {r["tweet_pos"]: r["label"] for r in res.data}
