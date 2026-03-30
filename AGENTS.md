# tweet-label-emo — Agent Guide

> Web app for labelling tweets with Plutchik's wheel of emotions.  
> The UI is in Serbian; code comments and this guide are in English.

---

## Project Overview

`tweet-label-emo` is a small Streamlit application used to manually annotate tweets from the `#NisamPrijavila` campaign with one of eight emotions (Plutchik's wheel) or a special label. Annotator progress is persisted to a Supabase PostgreSQL database.

### Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit entry point. Handles login, instructions, and the annotation UI. |
| `config.py` | Hard-coded constants: access codes, emotion lists, example tweets, label normalisation map. |
| `data_utils.py` | Loads `data.csv` and `emotions_chatgpt.jsonl`, builds stratified annotator samples, caches data with `@st.cache_data`. |
| `db.py` | Thin wrapper around the Supabase Python client for saving and loading annotations. |
| `data.csv` | Raw tweet dataset (`tweet_id`, `text`). |
| `emotions_chatgpt.jsonl` | ChatGPT-generated emotion labels for each tweet (`id_str`, `emotion`). |
| `supabase/migrations/20260325000000_create_annotations.sql` | Schema for the `annotations` table. |
| `.devcontainer/devcontainer.json` | Dev-container setup for GitHub Codespaces / VS Code. |
| `requirements.txt` | Python dependencies. |

---

## Technology Stack

- **Runtime:** Python 3.11
- **Web Framework:** Streamlit >= 1.35.0
- **Database:** Supabase (PostgreSQL + PostgREST)
- **Data Processing:** pandas >= 2.0.0
- **Development Environment:** VS Code Dev Container (Debian Bookworm-based Python image)

There is **no** `pyproject.toml`, `setup.py`, `Makefile`, Docker Compose file, or frontend build step.

---

## Architecture

The app is a single-file Streamlit script that uses **session-state based routing** to simulate three pages:

1. **Login** (`page_login`) — access-code entry.
2. **Instructions** (`page_instructions`) — emotion definitions and annotated examples.
3. **Annotation** (`page_annotation`) — tweet-by-tweet labelling interface.

### Sample Model

- **10 main annotators** (`anotator-01` … `anotator-10`) each get a stratified sample drawn from the full labelled dataset.
- **3 SCL (speed-controlled lab) annotators** (`scl-01` … `scl-03`) each get exactly 50 tweets and are shown a 15-minute countdown timer.
- Samples are generated deterministically via `random.Random(SEED)` in `data_utils.py`.

### Database Schema (`annotations`)

```sql
annotator_code  text
sample_idx      int
tweet_pos       int
tweet_id        text
label           text
created_at      timestamptz
```

Upserts use the unique constraint `(annotator_code, tweet_pos)` so re-labelling the same tweet overwrites the previous row.

### Security Model

- The Supabase table has **Row Level Security enabled** with anonymous `INSERT`, `SELECT`, and `UPDATE` policies (no auth required).
- Supabase credentials are read from **Streamlit secrets** (`st.secrets["supabase"]["url"]` and `["key"]`).
- The local secrets file `.streamlit/secrets.toml` is **gitignored**.

---

## Build and Run Commands

### Local Development

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'EOF'
[supabase]
url = "https://<your-project>.supabase.co"
key = "<your-anon-key>"
EOF

# 4. Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Dev Container / Codespaces

The `.devcontainer/devcontainer.json` automatically:

1. Installs packages from `requirements.txt` (and `packages.txt` if present).
2. Installs `streamlit` globally.
3. Starts the server on port `8501` with CORS and XSRF protection disabled:
   ```bash
   streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
   ```

---

## Code Style Guidelines

- **Type hints** are used for function signatures and module-level constants (e.g. `dict[str, int]`, `list[str]`).
- **String literals in the UI** are written in Serbian because the target users are Serbian-speaking annotators.
- **Comments and docstrings** are written in English.
- **Private helpers** in `app.py` are prefixed with an underscore (`_inject_enhancements`, `_examples_table`, `_tweet_card`).
- **Constants** are declared in `UPPER_SNAKE_CASE` inside `config.py` and `data_utils.py`.

There is no configured linter, formatter, or pre-commit hook in the repository.

---

## Testing Instructions

There are **no automated tests** in this repository.  
Manual smoke-test checklist when making changes:

1. **Login flow** — verify that valid access codes from `config.py` advance to the instructions page and invalid codes show an error.
2. **Sample loading** — confirm that each access code loads the expected stratified sample (check the sidebar counter).
3. **Annotation persistence** — label a tweet, refresh the page, and ensure the label is restored from Supabase.
4. **Re-labelling** — change an existing label and verify the database row is updated (not duplicated).
5. **Navigation** — test Previous / Next buttons and the auto-advance logic after clicking an emotion.
6. **Timer (SCL only)** — log in with `scl-01`, `scl-02`, or `scl-03` and confirm the 15-minute countdown appears in the top-right corner.
7. **Completion** — annotate every tweet in a small sample and verify the success message and `st.balloons()` appear.

---

## Security Considerations

- **Secrets:** Never commit `.streamlit/secrets.toml`. It is already listed in `.gitignore`.
- **Database RLS:** The Supabase policies allow anonymous writes. If the `anon` key is leaked, anyone can read or modify annotations. Treat the key as a public-but-sensitive credential and rotate it if exposed.
- **XSS:** `app.py` uses `unsafe_allow_html=True` and injects raw HTML/JS via `components.html`. Any change to user-facing markup should sanitise inputs to avoid script injection.
- **Data files:** `data.csv` and `emotions_chatgpt.jsonl` contain real tweet text. Do not expose them publicly if the dataset is under embargo or contains PII.

---

## Deployment Notes

The app is designed to run on **Streamlit Community Cloud** (or any Streamlit host):

1. Push the repository to GitHub.
2. Connect the repo to Streamlit Cloud.
3. Add the Supabase URL and key in the Streamlit Cloud "Secrets" manager.
4. Ensure the Supabase project is live and the `annotations` table matches the migration in `supabase/migrations/`.

There is no CI/CD pipeline, container registry, or serverless function layer beyond Streamlit + Supabase.
