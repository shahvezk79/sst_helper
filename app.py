"""
SST Decision Navigator — Streamlit frontend.

Run with:
    streamlit run app.py
"""

import html as html_lib
import logging
import re

import streamlit as st

from sst_navigator.pipeline import SSTNavigatorPipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SST Decision Navigator",
    page_icon="⚖️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "light_states" not in st.session_state:
    st.session_state.light_states = {
        "data": "off",
        "vectors": "off",
        "systems": "off",
    }
if "n_decisions" not in st.session_state:
    st.session_state.n_decisions = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pipeline() -> SSTNavigatorPipeline:
    """Return (or create) the singleton pipeline in session state."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = SSTNavigatorPipeline(
            dev_mode=True,
            generation_backend="mlx",
            fast_mode=False,
        )
    return st.session_state.pipeline


def _sublabel(name: str) -> str:
    """Return the small status text beneath each indicator light."""
    state = st.session_state.light_states[name]
    n = st.session_state.n_decisions
    if name == "data":
        return {"on": f"{n:,} decisions", "loading": "Downloading..."}.get(
            state, "Standby"
        )
    if name == "vectors":
        return {"on": f"{n:,} vectors", "loading": "Aligning..."}.get(
            state, "Standby"
        )
    return {"on": "All clear"}.get(state, "Awaiting subsystems")


def render_status_panel() -> str:
    """Build the HTML for the cockpit status indicator panel."""
    lights = [
        ("data", "DATA FEED"),
        ("vectors", "VECTOR INDEX"),
        ("systems", "SYSTEMS GO"),
    ]
    items = ""
    for key, label in lights:
        s = st.session_state.light_states[key]
        sub = _sublabel(key)
        lbl_cls = "ind-label-active" if s == "on" else ""
        sub_cls = "ind-sub-active" if s == "on" else ""
        items += (
            f'<div class="ind-group">'
            f'<div class="ind ind-{s}"></div>'
            f'<div class="ind-label {lbl_cls}">{label}</div>'
            f'<div class="ind-sub {sub_cls}">{sub}</div>'
            f"</div>"
        )
    return f'<div class="status-panel"><div class="status-row">{items}</div></div>'


def _md_to_html(text: str) -> str:
    """Minimal markdown-to-HTML for the LLM-generated case card."""
    safe = html_lib.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = safe.replace("\n", "<br>")
    return safe


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* ---- Layout ---- */
    .stApp { max-width: 1020px; margin: 0 auto; }

    /* ---- Status panel (cockpit lights) ---- */
    .status-panel {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.6rem 1rem;
        margin: 0.5rem 0 1.5rem 0;
    }
    .status-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
    .ind-group {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 140px;
    }
    .ind {
        width: 18px; height: 18px;
        border-radius: 50%;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .ind-off {
        background: #21262d;
        border: 2px solid #30363d;
    }
    .ind-loading {
        background: #f0883e;
        border: 2px solid #f0883e;
        animation: amber-pulse 0.7s ease-in-out infinite;
    }
    .ind-on {
        background: #3fb950;
        border: 2px solid #3fb950;
        box-shadow: 0 0 8px #3fb950, 0 0 18px rgba(63,185,80,0.35);
        animation: green-glow 2.5s ease-in-out infinite;
    }
    .ind-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        color: #484f58;
        text-transform: uppercase;
        font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
    }
    .ind-label-active { color: #3fb950; }
    .ind-sub {
        font-size: 0.62rem;
        color: #484f58;
        margin-top: 3px;
        font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
    }
    .ind-sub-active { color: #8b949e; }

    @keyframes amber-pulse {
        0%, 100% { box-shadow: 0 0 6px #f0883e, 0 0 14px rgba(240,136,62,0.25); }
        50%      { box-shadow: 0 0 10px #f0883e, 0 0 22px rgba(240,136,62,0.45); }
    }
    @keyframes green-glow {
        0%, 100% { box-shadow: 0 0 8px #3fb950, 0 0 18px rgba(63,185,80,0.35); }
        50%      { box-shadow: 0 0 12px #3fb950, 0 0 28px rgba(63,185,80,0.45),
                               0 0 40px rgba(63,185,80,0.12); }
    }

    /* ---- Hero ---- */
    .hero-bar {
        width: 48px; height: 3px;
        background: #3fb950;
        border-radius: 2px;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 1.9rem; font-weight: 800;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #656d76;
        margin-bottom: 0.3rem;
    }

    /* ---- Case card ---- */
    .case-card {
        background: #f6f8fa;
        border-left: 4px solid #3fb950;
        padding: 1.3rem 1.5rem;
        border-radius: 0 8px 8px 0;
        line-height: 1.7;
        font-size: 0.93rem;
    }

    /* ---- Result header inside expanders ---- */
    .rh {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.6rem;
    }
    .rh-rank {
        background: #0d1117; color: #3fb950;
        min-width: 34px; height: 34px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.82rem;
        font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
    }
    .rh-meta { flex: 1; }
    .rh-name { font-weight: 600; font-size: 0.92rem; margin-bottom: 1px; }
    .rh-date { font-size: 0.78rem; color: #656d76; }
    .rh-score { text-align: right; min-width: 80px; }
    .rh-pct {
        font-size: 0.92rem; font-weight: 700;
        font-family: 'SF Mono', 'Fira Code', 'Courier New', monospace;
    }
    .score-track {
        width: 80px; height: 4px;
        background: #d0d7de;
        border-radius: 2px;
        margin-top: 4px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #3fb950, #2ea043);
        border-radius: 2px;
    }
    .r-snippet {
        background: #f6f8fa;
        border: 1px solid #d8dee4;
        border-radius: 6px;
        padding: 0.9rem 1rem;
        font-size: 0.83rem;
        color: #424a53;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0.6rem 0;
    }
    .r-link {
        font-size: 0.83rem; color: #0969da;
        text-decoration: none; font-weight: 500;
    }
    .r-link:hover { text-decoration: underline; }

    /* ---- Disclaimer ---- */
    .disclaimer {
        background: #fff8c5;
        border: 1px solid #d4a72c;
        border-radius: 8px;
        padding: 0.65rem 1rem;
        font-size: 0.78rem;
        color: #57534e;
        line-height: 1.5;
        margin-bottom: 1rem;
    }

    /* ---- Green primary buttons ---- */
    button[data-testid="stBaseButton-primary"] {
        background-color: #238636 !important;
        border-color: #238636 !important;
        color: white !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #2ea043 !important;
        border-color: #2ea043 !important;
    }
    button[data-testid="stBaseButton-primary"]:disabled {
        background-color: #21262d !important;
        border-color: #30363d !important;
        color: #484f58 !important;
        opacity: 1 !important;
    }

    /* ---- Responsive ---- */
    @media (max-width: 640px) {
        .status-row { flex-direction: column; gap: 1.2rem; }
        .ind-group { min-width: auto; }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Header + Status panel  (rendered before sidebar so the placeholder exists
# when the init-button handler runs)
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-bar"></div>'
    '<div class="hero-title">SST Decision Navigator</div>'
    '<div class="hero-sub">'
    "AI-powered search across Social Security Tribunal decisions"
    "</div>",
    unsafe_allow_html=True,
)

status_placeholder = st.empty()
status_placeholder.markdown(render_status_panel(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — settings + initialization
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    dev_mode = st.toggle(
        "Development mode (500 rows)",
        value=True,
        help="Load only 500 decisions for faster startup.",
    )
    gen_backend = st.selectbox(
        "Summary backend",
        ["mlx", "openai", "gemini"],
        index=0,
        help="'mlx' = fully local. Others need an API key in env.",
    )
    fast_mode = st.toggle(
        "Fast mode",
        value=False,
        help="Fewer candidates, shorter inputs. Faster but slightly less accurate.",
    )

    st.divider()

    pipeline_ready = (
        "pipeline" in st.session_state and st.session_state.pipeline.is_ready
    )
    btn_label = "Re-initialize" if pipeline_ready else "Initialize Systems"

    if st.button(btn_label, type="primary", use_container_width=True):
        pipeline = get_pipeline()
        pipeline.dev_mode = dev_mode
        pipeline.set_generation_backend(gen_backend)
        pipeline.fast_mode = fast_mode

        try:
            # -- DATA FEED ------------------------------------------------
            st.session_state.light_states = {
                "data": "loading",
                "vectors": "off",
                "systems": "off",
            }
            status_placeholder.markdown(
                render_status_panel(), unsafe_allow_html=True
            )

            n = pipeline.load_data()
            st.session_state.n_decisions = n
            st.session_state.light_states["data"] = "on"
            status_placeholder.markdown(
                render_status_panel(), unsafe_allow_html=True
            )

            # -- VECTOR INDEX ---------------------------------------------
            st.session_state.light_states["vectors"] = "loading"
            status_placeholder.markdown(
                render_status_panel(), unsafe_allow_html=True
            )

            pipeline.build_index()
            st.session_state.light_states["vectors"] = "on"
            status_placeholder.markdown(
                render_status_panel(), unsafe_allow_html=True
            )

            # -- SYSTEMS GO -----------------------------------------------
            st.session_state.light_states["systems"] = "on"
            status_placeholder.markdown(
                render_status_panel(), unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Initialization failed: {e}")
            st.session_state.light_states = {
                "data": "off",
                "vectors": "off",
                "systems": "off",
            }
            st.session_state.n_decisions = 0
            status_placeholder.markdown(
                render_status_panel(), unsafe_allow_html=True
            )

    if pipeline_ready:
        st.success("Pipeline ready.")
    else:
        st.info("Press Initialize to start.")


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="disclaimer">'
    "<strong>Disclaimer:</strong> This is an educational research tool. "
    "It does not provide legal advice. Decisions retrieved here may not "
    "reflect the current state of the law. Consult a licensed professional "
    "or your local legal-aid clinic for legal help."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

col_a, col_b = st.columns([1, 3])

with col_a:
    benefit_type = st.selectbox(
        "Benefit type",
        [
            "General / All",
            "Employment Insurance",
            "Canada Pension Plan Disability",
            "Old Age Security",
        ],
        help="Filter by benefit area (planned feature).",
    )

with col_b:
    query = st.text_area(
        "Describe your situation in plain English",
        height=120,
        placeholder=(
            "Example: I was denied CPP disability benefits even though "
            "my doctor says I can't work due to chronic back pain and "
            "depression. I've been unable to hold any job for two years."
        ),
    )

is_ready = "pipeline" in st.session_state and st.session_state.pipeline.is_ready

search_clicked = st.button(
    "Search Decisions",
    type="primary",
    use_container_width=True,
    disabled=not (is_ready and query.strip()),
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if search_clicked and query.strip():
    pipeline = get_pipeline()
    pipeline.fast_mode = fast_mode

    with st.spinner("Running search pipeline..."):
        output = pipeline.search(query.strip())

    if output.results:
        st.markdown("#### Top Match")
        st.markdown(
            f'<div class="case-card">{_md_to_html(output.case_card)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### All Results")
        for r in output.results:
            pct = r.reranker_score * 100
            pct_s = f"{pct:.1f}%"

            with st.expander(
                f"#{r.rank}  {r.name}  \u2014  {r.date}  ({pct_s})"
            ):
                header_html = (
                    f'<div class="rh">'
                    f'<div class="rh-rank">#{r.rank}</div>'
                    f'<div class="rh-meta">'
                    f'<div class="rh-name">{html_lib.escape(r.name)}</div>'
                    f'<div class="rh-date">{html_lib.escape(r.date)}</div>'
                    f"</div>"
                    f'<div class="rh-score">'
                    f'<div class="rh-pct">{pct_s}</div>'
                    f'<div class="score-track">'
                    f'<div class="score-fill" style="width:{pct}%"></div>'
                    f"</div></div></div>"
                )
                st.markdown(header_html, unsafe_allow_html=True)

                if r.url:
                    st.markdown(
                        f'<a class="r-link" href="{html_lib.escape(r.url)}" '
                        f'target="_blank">View official decision &rarr;</a>',
                        unsafe_allow_html=True,
                    )

                snippet = r.snippet + (
                    "..." if len(r.full_text) > len(r.snippet) else ""
                )
                st.markdown(
                    f'<div class="r-snippet">{html_lib.escape(snippet)}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No matching decisions were found for this query.")
