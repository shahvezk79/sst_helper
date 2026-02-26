"""
SST Decision Navigator — Streamlit frontend.

Run with:
    streamlit run app.py
"""

import logging
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
# Custom CSS for a clean, accessible look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { max-width: 960px; margin: 0 auto; }
    .case-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1.2rem 1.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    .score-badge {
        display: inline-block;
        background: #1f77b4;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .disclaimer {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state helpers
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


# ---------------------------------------------------------------------------
# Sidebar — initialisation controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    dev_mode = st.toggle(
        "Development mode (500 rows)",
        value=True,
        help=(
            "When enabled, only the first 500 decisions are loaded for faster "
            "startup and iteration. Turn off to index the full dataset."
        ),
    )
    gen_backend = st.selectbox(
        "Summary backend",
        ["mlx", "openai", "gemini"],
        index=0,
        help="'mlx' runs fully local. 'openai'/'gemini' need an API key in env.",
    )

    fast_mode = st.toggle(
        "Fast mode (lower latency, slightly lower accuracy)",
        value=False,
        help=(
            "Uses fewer retrieval candidates and shorter model inputs to reduce "
            "embedding/search/generation latency. May reduce ranking and summary quality."
        ),
    )
    if fast_mode:
        st.warning("Fast mode is ON: results may be slightly less accurate.")

    st.divider()

    if st.button("Load data & build index", type="primary", use_container_width=True):
        pipeline = get_pipeline()
        pipeline.dev_mode = dev_mode
        pipeline.set_generation_backend(gen_backend)
        pipeline.fast_mode = fast_mode

        with st.status("Initialising pipeline…", expanded=True) as status:
            # Step 1: load data
            st.write("Downloading SST decisions…")
            n_rows = pipeline.load_data()
            st.write(f"Loaded **{n_rows}** decisions.")

            # Step 2: download and align precomputed embeddings
            st.write("Downloading precomputed embeddings from HuggingFace…")
            pipeline.build_index(progress_callback=None)
            status.update(label="Pipeline ready!", state="complete")

    if "pipeline" in st.session_state and st.session_state.pipeline.is_ready:
        st.success("Pipeline is ready.")
    else:
        st.info("Load data to get started.")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("SST Decision Navigator")

st.markdown(
    '<div class="disclaimer">'
    "<strong>Disclaimer:</strong> This is an educational research tool. "
    "It does <em>not</em> provide legal advice. Tribunal decisions retrieved "
    "here may not reflect the current state of the law. If you need legal "
    "help, consult a licensed professional or your local legal-aid clinic."
    "</div>",
    unsafe_allow_html=True,
)

# -- Inputs ----------------------------------------------------------------

col1, col2 = st.columns([1, 3])

with col1:
    benefit_type = st.selectbox(
        "Benefit type",
        [
            "General / All",
            "Employment Insurance",
            "Canada Pension Plan Disability",
            "Old Age Security",
        ],
        help="Filter by benefit area (filtering is a planned feature).",
    )

with col2:
    query = st.text_area(
        "Describe the facts of your situation in plain English",
        height=120,
        placeholder=(
            "Example: I was denied CPP disability benefits even though "
            "my doctor says I can't work due to chronic back pain and "
            "depression. I've been unable to hold any job for two years."
        ),
    )

search_clicked = st.button(
    "Find Similar Cases",
    type="primary",
    use_container_width=True,
    disabled=not (
        "pipeline" in st.session_state
        and st.session_state.pipeline.is_ready
        and query.strip()
    ),
)

# -- Results ---------------------------------------------------------------

if search_clicked and query.strip():
    pipeline = get_pipeline()

    pipeline.fast_mode = fast_mode

    with st.spinner("Searching and analysing decisions…"):
        output = pipeline.search(query.strip())

    # Case card for the top result
    st.subheader("Case Card — Top Match")
    st.markdown(
        f'<div class="case-card">{output.case_card}</div>',
        unsafe_allow_html=True,
    )

    # Expandable details for top 3
    st.subheader("Top Matching Decisions")

    for r in output.results:
        score_pct = f"{r.reranker_score * 100:.1f}%"
        with st.expander(
            f"#{r.rank}  {r.name}  —  {r.date}  "
            f"(confidence {score_pct})"
        ):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**Case:** {r.name}")
                st.markdown(f"**Date:** {r.date}")
            with cols[1]:
                st.markdown(
                    f'<span class="score-badge">{score_pct}</span>',
                    unsafe_allow_html=True,
                )

            if r.url:
                st.markdown(f"[View official decision]({r.url})")

            st.markdown("---")
            st.text(r.snippet + ("…" if len(r.full_text) > len(r.snippet) else ""))
