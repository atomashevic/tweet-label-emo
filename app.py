import json
import time

import streamlit as st
import streamlit.components.v1 as components

from config import ACCESS_CODES, EMOTIONS, EXAMPLES, SCL_CODES, SPECIAL_LABELS
from data_utils import get_sample
from db import load_annotations, save_annotation

st.set_page_config(page_title="Anotacija tvitova", page_icon="🏷️", layout="centered")

# ── Session state init ────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "page": "login",
        "code": None,
        "sample_idx": None,
        "annotations": {},
        "current_pos": 0,
        "annotation_start_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

_EMOTION_COLORS: dict[str, str] = {
    "bes":          "#ffe0e0",
    "tuga":         "#dde8ff",
    "strah":        "#d6f0e0",
    "poverenje":    "#d0f5e8",
    "gađenje":      "#e8d6f5",
    "radost":       "#fff9d0",
    "iščekivanje":  "#ffe8cc",
    "iznenađenje":  "#d0f0ff",
}

# Darker shade for selected state
_EMOTION_COLORS_SELECTED: dict[str, str] = {
    "bes":          "#ffaaaa",
    "tuga":         "#99b8ff",
    "strah":        "#90d8a8",
    "poverenje":    "#80e0b8",
    "gađenje":      "#c890e8",
    "radost":       "#f0d840",
    "iščekivanje":  "#ffb060",
    "iznenađenje":  "#70c8f0",
}


def _inject_enhancements(current_label: str | None, start_ms: int | None, show_timer: bool) -> None:
    """Inject button color styling and optional countdown timer via components.html."""
    timer_code = ""
    if show_timer and start_ms is not None:
        timer_code = f"""
        // Create timer div in parent page if missing
        let timerEl = doc.getElementById('cdtimer');
        if (!timerEl) {{
            timerEl = doc.createElement('div');
            timerEl.id = 'cdtimer';
            Object.assign(timerEl.style, {{
                position: 'fixed', top: '56px', right: '20px', zIndex: '9999',
                background: '#fff', border: '1.5px solid #ddd', borderRadius: '10px',
                padding: '6px 16px', fontSize: '1.35rem', fontWeight: '700',
                fontFamily: 'monospace', boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
                color: '#333', letterSpacing: '2px',
            }});
            doc.body.appendChild(timerEl);
        }}

        // Define tick on parent window so it survives iframe teardown on rerender
        window.parent._cdStartMs = {start_ms};
        window.parent._cdTick = function() {{
            const el = window.parent.document.getElementById('cdtimer');
            if (!el) return;
            const remaining = Math.max(0, 15 * 60 * 1000 - (Date.now() - window.parent._cdStartMs));
            const m = Math.floor(remaining / 60000);
            const s = Math.floor((remaining % 60000) / 1000);
            el.textContent = String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
            el.style.color = remaining === 0 ? '#cc0000' : remaining < 60000 ? '#e06000' : '#333';
            el.style.borderColor = remaining < 60000 ? '#e06000' : '#ddd';
        }};
        window.parent._cdTick();

        // Clear old interval and restart on parent window (survives iframe recreation)
        if (window.parent._cdInterval) window.parent.clearInterval(window.parent._cdInterval);
        window.parent._cdInterval = window.parent.setInterval(window.parent._cdTick, 1000);
        """

    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            const HOVER    = {json.dumps(_EMOTION_COLORS)};
            const SELECTED = {json.dumps(_EMOTION_COLORS_SELECTED)};
            const currentLabel = {json.dumps(current_label or "")};

            {timer_code}

            function applyStyles() {{
                doc.querySelectorAll('button').forEach(btn => {{
                    const text = btn.innerText.trim();
                    const key  = text.toLowerCase();

                    if (HOVER[key] !== undefined) {{
                        // ── Emotion button ──
                        const sel = key === currentLabel.toLowerCase() && currentLabel !== '';
                        if (sel) {{
                            btn.style.backgroundColor = SELECTED[key];
                            btn.style.borderColor     = 'rgba(0,0,0,0.28)';
                            btn.style.fontWeight      = '700';
                            btn.style.color           = '#111';
                            btn.onmouseenter = () => {{ btn.style.backgroundColor = SELECTED[key]; }};
                            btn.onmouseleave = () => {{ btn.style.backgroundColor = SELECTED[key]; }};
                        }} else {{
                            btn.style.backgroundColor = '';
                            btn.style.borderColor     = '';
                            btn.style.fontWeight      = '';
                            btn.style.color           = '';
                            btn.onmouseenter = () => {{ btn.style.backgroundColor = HOVER[key]; }};
                            btn.onmouseleave = () => {{ btn.style.backgroundColor = '';           }};
                        }}

                    }} else if (text === 'Emocionalno neutralno' || text === 'Ne mogu da razumem') {{
                        // ── Special label button ──
                        const sel = text === currentLabel;
                        const bg  = sel ? '#444' : '#888';
                        const hov = sel ? '#333' : '#666';
                        btn.style.backgroundColor = bg;
                        btn.style.color           = '#fff';
                        btn.style.borderColor     = sel ? '#222' : '#666';
                        btn.style.fontWeight      = sel ? '700' : '';
                        btn.onmouseenter = () => {{ btn.style.backgroundColor = hov; }};
                        btn.onmouseleave = () => {{ btn.style.backgroundColor = bg;  }};
                    }}
                }});
            }}

            applyStyles();
            new MutationObserver(applyStyles).observe(doc.body, {{ childList: true, subtree: true }});
        }})();
        </script>
        """,
        height=0,
    )


def _examples_table():
    rows_html = ""
    for e in EXAMPLES:
        bg = _EMOTION_COLORS.get(e["emotion"], "#f5f5f5")
        rows_html += f"""
        <tr style="background:{bg};">
          <td style="padding:10px 14px;font-weight:600;white-space:nowrap;vertical-align:top;">{e["emotion"]}</td>
          <td style="padding:10px 14px;vertical-align:top;">{e["tweet"]}</td>
          <td style="padding:10px 14px;color:#444;vertical-align:top;">{e["explanation"]}</td>
        </tr>"""
    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem;line-height:1.5;">
          <thead>
            <tr style="background:#e0e0e0;">
              <th style="padding:10px 14px;text-align:left;white-space:nowrap;">Emocija</th>
              <th style="padding:10px 14px;text-align:left;">Tvit</th>
              <th style="padding:10px 14px;text-align:left;">Objašnjenje</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def _tweet_card(text: str):
    st.markdown(
        f"""
        <div style="
            background:#f0f2f6;
            border-radius:12px;
            padding:20px 24px;
            font-size:1.15rem;
            line-height:1.6;
            margin-bottom:16px;
        ">{text}</div>
        """,
        unsafe_allow_html=True,
    )


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_login():
    st.title("Anotacija tvitova — #NisamPrijavila")
    st.markdown("Unesite vaš pristupni kod da biste počeli.")
    code_input = st.text_input("Pristupni kod", type="password", key="login_input")
    if st.button("Prijavi se", use_container_width=True):
        code = code_input.strip()
        if code in ACCESS_CODES:
            sample_idx = ACCESS_CODES[code]
            annotations = load_annotations(code)
            sample = get_sample(sample_idx)
            n = len(sample)
            current_pos = next(
                (i for i in range(n) if i not in annotations), n
            )
            st.session_state.code = code
            st.session_state.sample_idx = sample_idx
            st.session_state.annotations = annotations
            st.session_state.current_pos = current_pos
            st.session_state.page = "consent"
            st.rerun()
        else:
            st.error("Nevažeći pristupni kod. Pokušajte ponovo.")


def page_consent():
    st.title("Saglasnost za ispitanike")
    st.markdown(
        """
Zamolili bismo Vas da učestvujete u ovom istraživanju čiji je cilj procena tvitova **#nisamprijavila**.
Vaše učešće je potpuno **anonimno** i od vas se neće tražiti nikakvi lični podaci, niti demografski podaci.

Vaš zadatak će biti da procenite **50 tvitova** u odnosu na **dominantan emocionalni ton**,
u skladu sa detaljnijim uputstvom koje ćete dobiti. Procena tvitova će trajati **maksimalno 20-ak minuta**.

Učešće u ovom istraživanju je **dobrovoljno** i od njega možete odustati u bilo kojem trenutku,
bez ikakvih posledica. Učešće u istraživanju je bez materijalne ili druge nadoknade.

Istraživanje vodi interdisciplinarni tim, a vodeću grupu istraživača čine saradnici sa Odseka za psihologiju
Filozofskog fakulteta u Novom Sadu i Instituta za fiziku u Beogradu.
Ukoliko imate pitanja u vezi s istraživanjem, možete se obratiti na mejl **bojana.dinic@ff.uns.ac.rs**.
        """
    )
    if st.button("Dajem saglasnost, nastavi dalje →", use_container_width=True):
        st.session_state.page = "instructions"
        st.rerun()


def page_instructions():
    st.title("Uputstvo za anotaciju")
    st.markdown(
        """
Ovo su primeri tvitova koji su prikupljeni u određenom periodu preko Twitter/X platforme,
a koji sadrže **#nisamprijavila**. ID tvita postoji zarad naše evidencije,
a u okviru tvitova su izostavljeni nazivi korisnika, tj. stoji `@[user]`.

| tweet_id | text |
|---|---|
| 1481522225534623744 | Nekada daaavno nosili su bedževe sa natpisom "Čedo oženi me". <br><br>Ja od danas nosim ovaj: <br><br>#NisamPrijavila |
| 1481414310836551681 | Pisala sam o #nisamprijavila <br><br>Trenutno je objavljeno na radnoj verziji platforme. Ne zamerite. Inspiracija ne bira. Već neko vreme planiram da objavljujem svoja pisanija, ali eto tek sada skupljam hrabrosti. :) |
| 1481406780102090756 | @[user] Ovo je baš za hashtag #NisamPrijavila |

---

**Uputstvo**

Vaš zadatak je da svakom tvitu dodelite **jednu od 8 emocija** koja dominira u tvitu.

Ukoliko tvit ne izražava nikakvu emociju, odaberite **Emocionalno neutralno**.
Ukoliko ne možete da razumete tvit, odaberite **Ne mogu da razumem**.

**Emocije:**
- **poverenje** — osećaj sigurnosti, oslanjanja na nekoga
- **bes** — gnev, ljutnja, ogorčenost
- **tuga** — žalost, bol, gubitak
- **iznenađenje** — šok, zapanjenost, neočekivanost
- **strah** — uznemirenost, bojazan, pretnja
- **gađenje** — odbojnost, odvratnost, moralno negodovanje
- **radost** — sreća, olakšanje, nada
- **iščekivanje** — napetost, očekivanje, praćenje razvoja situacije

Ispod su primeri za svaku emociju:
        """
    )
    _examples_table()
    if st.button("Nastavi na anotaciju →", use_container_width=True):
        st.session_state.page = "annotation"
        if st.session_state.annotation_start_time is None:
            st.session_state.annotation_start_time = int(time.time() * 1000)
        st.rerun()


def page_annotation():
    sample_idx = st.session_state.sample_idx
    code = st.session_state.code
    annotations = st.session_state.annotations
    sample = get_sample(sample_idx)
    n = len(sample)

    # Sidebar
    with st.sidebar:
        st.markdown(f"**Anotator:** `{code}`")
        st.markdown(f"**Označeno:** {len(annotations)} / {n}")
        if st.button("↩ Uputstvo"):
            st.session_state.page = "instructions"
            st.rerun()

    current_pos = st.session_state.current_pos

    # Completion check
    if len(annotations) >= n:
        st.balloons()
        st.success("Hvala! Završili ste anotaciju.")
        st.markdown(f"Označili ste svih **{n}** tvitova.")
        return

    # Clamp position
    current_pos = max(0, min(current_pos, n - 1))
    st.session_state.current_pos = current_pos

    # Progress
    st.progress(len(annotations) / n)
    st.markdown(f"**Tvit {current_pos + 1} od {n}**")

    # Collapsible examples
    with st.expander("Primeri emocija ▾"):
        _examples_table()

    # Tweet card
    row = sample.iloc[current_pos]
    tweet_text = row["text"]
    tweet_id = row["tweet_id"]
    _tweet_card(tweet_text)

    # Current label
    current_label = annotations.get(current_pos)
    if current_label:
        st.success(f"Trenutna oznaka: **{current_label}**")

    # Emotion buttons — 2 rows × 4 cols
    st.markdown("**Izaberite emociju:**")
    cols1 = st.columns(4)
    for i, emotion in enumerate(EMOTIONS[:4]):
        with cols1[i]:
            if st.button(emotion, key=f"emo_{current_pos}_{emotion}", use_container_width=True):
                save_annotation(code, sample_idx, current_pos, tweet_id, emotion)
                annotations[current_pos] = emotion
                st.session_state.annotations = annotations
                next_pos = next(
                    (j for j in range(current_pos + 1, n) if j not in annotations),
                    next((j for j in range(n) if j not in annotations), n),
                )
                st.session_state.current_pos = next_pos
                st.rerun()

    cols2 = st.columns(4)
    for i, emotion in enumerate(EMOTIONS[4:]):
        with cols2[i]:
            if st.button(emotion, key=f"emo_{current_pos}_{emotion}", use_container_width=True):
                save_annotation(code, sample_idx, current_pos, tweet_id, emotion)
                annotations[current_pos] = emotion
                st.session_state.annotations = annotations
                next_pos = next(
                    (j for j in range(current_pos + 1, n) if j not in annotations),
                    next((j for j in range(n) if j not in annotations), n),
                )
                st.session_state.current_pos = next_pos
                st.rerun()

    # Special labels
    st.markdown("**Posebne oznake:**")
    spec_cols = st.columns(2)
    for i, special in enumerate(SPECIAL_LABELS):
        with spec_cols[i]:
            if st.button(special, key=f"spec_{current_pos}_{special}", use_container_width=True):
                save_annotation(code, sample_idx, current_pos, tweet_id, special)
                annotations[current_pos] = special
                st.session_state.annotations = annotations
                next_pos = next(
                    (j for j in range(current_pos + 1, n) if j not in annotations),
                    next((j for j in range(n) if j not in annotations), n),
                )
                st.session_state.current_pos = next_pos
                st.rerun()

    # Navigation arrows
    st.markdown("---")
    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button("← Prethodni", disabled=(current_pos == 0), use_container_width=True):
            st.session_state.current_pos = current_pos - 1
            st.rerun()
    with nav_cols[1]:
        if st.button("Sledeći →", disabled=(current_pos == n - 1), use_container_width=True):
            st.session_state.current_pos = current_pos + 1
            st.rerun()

    # ── Inject button colors + timer (runs in sandboxed iframe, targets parent DOM) ──
    _inject_enhancements(
        current_label=current_label,
        start_ms=st.session_state.annotation_start_time,
        show_timer=(code in SCL_CODES),
    )


# ── Router ────────────────────────────────────────────────────────────────────

page = st.session_state.page
if page == "login":
    page_login()
elif page == "consent":
    page_consent()
elif page == "instructions":
    page_instructions()
elif page == "annotation":
    page_annotation()
