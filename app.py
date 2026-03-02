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

MAX_CASE_CARD_CACHE_ENTRIES = 20

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
if "last_output" not in st.session_state:
    st.session_state.last_output = None
if "case_card_cache" not in st.session_state:
    st.session_state.case_card_cache = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pipeline() -> SSTNavigatorPipeline:
    """Return (or create) the singleton pipeline in session state."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = SSTNavigatorPipeline(
            dev_mode=False,
            generation_backend="mlx",
            compute_mode="cloud",
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


def _generation_cache_key(base_key: str, backend: str, fast_mode: bool) -> str:
    """Build a cache key tied to generation-related settings."""
    return f"{base_key}|backend={backend}|fast_mode={fast_mode}"


def _cache_case_card(cache_key: str, value: str) -> None:
    """Store a case-card in bounded cache (FIFO eviction)."""
    cache = st.session_state.case_card_cache
    cache[cache_key] = value
    while len(cache) > MAX_CASE_CARD_CACHE_ENTRIES:
        oldest = next(iter(cache))
        del cache[oldest]


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* ---- Global ---- */
    .stApp {
        max-width: 1080px;
        margin: 0 auto;
    }

    /* ---- Status panel ---- */
    .status-panel {
        background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(148, 163, 184, 0.08);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin: 0.75rem 0 2rem 0;
        box-shadow:
            0 4px 24px rgba(0, 0, 0, 0.12),
            0 1px 4px rgba(0, 0, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.04);
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
        min-width: 150px;
    }
    .ind {
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-bottom: 14px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .ind-off {
        background: #334155;
        border: 2px solid #475569;
    }
    .ind-loading {
        background: #f59e0b;
        border: 2px solid #f59e0b;
        animation: pulse-loading 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    .ind-on {
        background: #10b981;
        border: 2px solid #10b981;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.6),
                    0 0 24px rgba(16, 185, 129, 0.25);
        animation: glow-active 3s ease-in-out infinite;
    }
    .ind-label {
        font-size: 0.67rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #64748b;
        text-transform: uppercase;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .ind-label-active { color: #34d399; }
    .ind-sub {
        font-size: 0.7rem;
        color: #475569;
        margin-top: 4px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 500;
    }
    .ind-sub-active { color: #94a3b8; }

    @keyframes pulse-loading {
        0%, 100% {
            box-shadow: 0 0 4px rgba(245, 158, 11, 0.4),
                        0 0 14px rgba(245, 158, 11, 0.15);
            opacity: 1;
        }
        50% {
            box-shadow: 0 0 8px rgba(245, 158, 11, 0.6),
                        0 0 22px rgba(245, 158, 11, 0.3);
            opacity: 0.7;
        }
    }
    @keyframes glow-active {
        0%, 100% {
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.5),
                        0 0 20px rgba(16, 185, 129, 0.2);
        }
        50% {
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.6),
                        0 0 32px rgba(16, 185, 129, 0.3);
        }
    }

    /* ---- Hero ---- */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg,
            rgba(16, 185, 129, 0.08),
            rgba(16, 185, 129, 0.03));
        border: 1px solid rgba(16, 185, 129, 0.18);
        border-radius: 100px;
        padding: 5px 14px;
        font-size: 0.73rem;
        font-weight: 600;
        color: #10b981;
        letter-spacing: 0.03em;
        margin-bottom: 0.9rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.035em;
        line-height: 1.15;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .hero-sub {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 0.25rem;
        line-height: 1.55;
        font-weight: 400;
        max-width: 600px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Section headers ---- */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.75rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.6rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg,
            #e2e8f0 0%, rgba(226, 232, 240, 0) 100%);
        margin: 1.75rem 0;
        border: none;
    }

    /* ---- Case card ---- */
    .case-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #f8fafc 100%);
        border-left: 3px solid #10b981;
        padding: 1.5rem 1.75rem;
        border-radius: 0 12px 12px 0;
        line-height: 1.75;
        font-size: 0.92rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04),
                    0 1px 2px rgba(0, 0, 0, 0.02);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Result header inside expanders ---- */
    .rh {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    .rh-rank {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        color: #34d399;
        min-width: 36px; height: 36px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.8rem;
        font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
    }
    .rh-meta { flex: 1; }
    .rh-name {
        font-weight: 600; font-size: 0.92rem; margin-bottom: 2px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .rh-date {
        font-size: 0.78rem; color: #64748b;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .rh-score { text-align: right; min-width: 80px; }
    .rh-pct {
        font-size: 0.88rem; font-weight: 700;
        font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
        color: #1e293b;
    }
    .score-track {
        width: 80px; height: 4px;
        background: #e2e8f0;
        border-radius: 4px;
        margin-top: 5px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 4px;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .r-snippet {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.7;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin: 0.75rem 0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .r-link {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        font-size: 0.84rem;
        color: #3b82f6;
        text-decoration: none;
        font-weight: 500;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        transition: color 0.2s ease;
    }
    .r-link:hover {
        color: #2563eb;
        text-decoration: underline;
    }

    /* ---- Disclaimer ---- */
    .disclaimer {
        background: linear-gradient(135deg, #fffbeb 0%, #fefce8 100%);
        border: 1px solid #fde68a;
        border-radius: 10px;
        padding: 0.75rem 1.15rem;
        font-size: 0.8rem;
        color: #92400e;
        line-height: 1.55;
        margin-bottom: 1.25rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Primary buttons ---- */
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(145deg, #10b981, #059669) !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 1px 3px rgba(16, 185, 129, 0.25),
                    0 1px 2px rgba(0, 0, 0, 0.06) !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(145deg, #059669, #047857) !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3),
                    0 2px 4px rgba(0, 0, 0, 0.08) !important;
        transform: translateY(-1px) !important;
    }
    button[data-testid="stBaseButton-primary"]:active {
        transform: translateY(0) !important;
        box-shadow: 0 1px 2px rgba(16, 185, 129, 0.2) !important;
    }
    button[data-testid="stBaseButton-primary"]:disabled {
        background: #e2e8f0 !important;
        border: none !important;
        color: #94a3b8 !important;
        opacity: 1 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stHeading"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.02em;
    }

    /* ---- Expander ---- */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03) !important;
        transition: box-shadow 0.2s ease !important;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
    }

    /* ---- Text area ---- */
    [data-testid="stTextArea"] textarea {
        border-radius: 8px !important;
        border-color: #e2e8f0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important;
    }

    /* ---- Select box ---- */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 8px !important;
        border-color: #e2e8f0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ---- Responsive ---- */
    @media (max-width: 640px) {
        .status-row { flex-direction: column; gap: 1.5rem; }
        .ind-group { min-width: auto; }
        .hero-title { font-size: 1.75rem; }
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
    '<div class="hero-badge">AI-Powered Legal Research</div>'
    '<div class="hero-title">SST Decision Navigator</div>'
    '<div class="hero-sub">'
    "Search and analyze Social Security Tribunal decisions with semantic AI "
    "&mdash; find the cases most relevant to your situation."
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

    compute_mode = st.radio(
        "Compute",
        ["Local (MLX)", "Cloud (DeepInfra)"],
        index=0,
        help=(
            "**Local** runs reranking and generation on Apple Silicon. "
            "**Cloud** sends both to DeepInfra APIs (set DEEPINFRA_API_KEY)."
        ),
    )
    compute_key = "local" if "Local" in compute_mode else "cloud"

    if compute_key == "local":
        gen_backend = st.selectbox(
            "Summary backend",
            ["mlx", "openai", "gemini"],
            index=0,
            help="'mlx' = fully local. Others need an API key in env.",
        )
    else:
        gen_backend = "deepinfra"

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
        pipeline.set_compute_mode(compute_key)
        pipeline.set_generation_backend(gen_backend)
        pipeline.fast_mode = fast_mode

        st.session_state.last_output = None
        st.session_state.case_card_cache.clear()

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
    "<strong>Disclaimer</strong> &mdash; This is an educational research tool "
    "and does not constitute legal advice. Decisions retrieved may not "
    "reflect the current state of the law. Please consult a licensed "
    "professional or your local legal-aid clinic for legal guidance."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="section-divider"></div>'
    '<div class="section-label">Search</div>',
    unsafe_allow_html=True,
)

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

    with st.spinner("Running retrieval + reranking..."):
        output = pipeline.search(query.strip(), include_case_card=False)
    st.session_state.last_output = output

output = st.session_state.last_output
if output is not None:
    if output.results:
        st.markdown(
            '<div class="section-divider"></div>'
            '<div class="section-label">Results</div>'
            '<div class="section-title">Top Match</div>',
            unsafe_allow_html=True,
        )
        top_result = output.results[0]
        base_cache_key = top_result.url or f"rank-1:{top_result.name}:{top_result.date}"
        cache_key = _generation_cache_key(base_cache_key, gen_backend, fast_mode)
        cached_card = st.session_state.case_card_cache.get(cache_key)

        if cached_card is None:
            st.caption("Summary is generated on demand to keep search fast.")
            if st.button("Generate summary for top match", type="primary"):
                pipeline = get_pipeline()
                pipeline.set_generation_backend(gen_backend)
                pipeline.fast_mode = fast_mode
                with st.spinner("Generating case-card summary..."):
                    generated = pipeline.generate_case_card(top_result.full_text)
                _cache_case_card(cache_key, generated)
                st.rerun()
        else:
            st.markdown(
                f'<div class="case-card">{_md_to_html(cached_card)}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-title" style="margin-top: 1.5rem;">'
            "All Results</div>",
            unsafe_allow_html=True,
        )
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
