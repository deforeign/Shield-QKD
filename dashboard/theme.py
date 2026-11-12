"""
Visual identity for the ShieldQKD console.

An "optical bench" instrument panel: deep ink-navy surfaces, hairline
structure, IBM Plex (the typeface family of IBM / Qiskit).  Colour is used
only where it carries meaning:
    * green / amber / red  -> circuit-breaker verdicts, nothing else
    * blue / violet        -> the two measurement bases (rectilinear / diagonal)
The single bold element is the verdict readout; everything else stays quiet.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

INK = "#151C2F"          # page background
PANEL = "#1E2740"        # raised surfaces
HAIRLINE = "#2E3A5C"     # structure lines
TEXT = "#E6EAF4"
MUTED = "#9AA4BF"
DIM = "#5B6A92"

PROCEED = "#3FBF8A"
CAUTION = "#E7B547"
ABORT = "#F0605D"
RECT = "#7FB3FF"         # rectilinear basis / primary accent
DIAG = "#C79BFF"         # diagonal basis

DECISION_COLOR = {"PROCEED": PROCEED, "CAUTION": CAUTION, "WARN": CAUTION,
                  "ABORT_ATTACK": ABORT, "ABORT_MITM": ABORT}

FONT = "IBM Plex Sans, system-ui, sans-serif"
CONDENSED = "IBM Plex Sans Condensed, IBM Plex Sans, system-ui, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&display=swap');

.block-container {{ padding-top: 2.2rem; max-width: 1360px; }}
h1, h2, h3, h4 {{ font-family: {CONDENSED}; letter-spacing: 0.005em; }}

/* wordmark */
.sq-head {{ display: flex; align-items: baseline; gap: 1.1rem; flex-wrap: wrap;
           border-bottom: 1px solid {HAIRLINE}; padding-bottom: 0.9rem; margin-bottom: 0.4rem; }}
.sq-head .mark {{ font-family: {CONDENSED}; font-weight: 700; font-size: 2.15rem; line-height: 1;
                 color: {TEXT}; letter-spacing: -0.01em; }}
.sq-head .what {{ color: {MUTED}; font-size: 0.95rem; max-width: 62ch; }}

/* tabs: quiet underline, no pills */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {HAIRLINE}; }}
.stTabs [data-baseweb="tab"] {{ padding: 0.55rem 0; font-weight: 500; color: {MUTED}; }}
.stTabs [aria-selected="true"] {{ color: {TEXT}; }}

/* the verdict readout - the one bold element on the page */
.sq-verdict {{ display: grid; grid-template-columns: 6px 1fr; background: {PANEL};
              border: 1px solid {HAIRLINE}; border-radius: 6px; overflow: hidden; margin: 0.6rem 0 1.1rem; }}
.sq-verdict .beam {{ background: var(--state); }}
.sq-verdict .body {{ padding: 1.1rem 1.4rem 1.2rem; }}
.sq-verdict .kicker {{ color: {MUTED}; font-size: 0.85rem; }}
.sq-verdict .word {{ font-family: {CONDENSED}; font-weight: 700; font-size: clamp(2.2rem, 5vw, 3.6rem);
                    line-height: 1.02; color: var(--state); letter-spacing: 0.01em; margin: 0.15rem 0 0.35rem; }}
.sq-verdict .why {{ font-size: 1.08rem; color: {TEXT}; max-width: 70ch; }}
.sq-verdict dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1.2rem;
                 margin: 0.9rem 0 0; font-size: 0.93rem; }}
.sq-verdict dt {{ color: {MUTED}; }}
.sq-verdict dd {{ margin: 0; color: {TEXT}; }}

/* status strip: one instrument row split by hairlines, not five cards */
.sq-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            border-top: 1px solid {HAIRLINE}; border-bottom: 1px solid {HAIRLINE}; margin: 0.2rem 0 0.4rem; }}
.sq-strip .cell {{ padding: 0.7rem 1rem 0.75rem; border-left: 1px solid {HAIRLINE}; }}
.sq-strip .cell:first-child {{ border-left: none; padding-left: 0; }}
.sq-strip .lab {{ color: {MUTED}; font-size: 0.82rem; }}
.sq-strip .val {{ font-family: {CONDENSED}; font-weight: 600; font-size: 1.45rem; color: {TEXT};
                 font-variant-numeric: tabular-nums; line-height: 1.25; }}
.sq-strip .note {{ color: {MUTED}; font-size: 0.8rem; }}

/* session stage tracker: the six real stages, in order */
.sq-stages {{ display: flex; align-items: center; flex-wrap: wrap; gap: 0.35rem 0; font-size: 0.86rem;
             color: {MUTED}; padding: 0.55rem 0 0.9rem; }}
.sq-stages .st {{ white-space: nowrap; }}
.sq-stages .st.done {{ color: {TEXT}; }}
.sq-stages .st.final {{ color: {TEXT}; font-weight: 600; }}
.sq-stages .st.stopped {{ color: {ABORT}; font-weight: 600; }}
.sq-stages .st.ready {{ color: {RECT}; font-weight: 600; }}
.sq-stages .st.blocked, .sq-stages .st.skipped {{ color: {DIM}; text-decoration: line-through; }}
.sq-stages .link {{ flex: 1 1 18px; min-width: 14px; height: 1px; background: {HAIRLINE}; margin: 0 0.7rem; }}

/* compact verdict for the console's left column */
.sq-verdict.compact .body {{ padding: 0.95rem 1.2rem 1rem; }}
.sq-verdict.compact .word {{ font-size: clamp(2rem, 3.4vw, 2.7rem); }}
.sq-verdict.compact .why {{ font-size: 1rem; }}

/* instrument list: label left, reading right, hairline between rows */
.sq-instr {{ border-top: 1px solid {HAIRLINE}; margin: 0 0 1.3rem; font-size: 0.92rem; }}
.sq-instr .row {{ display: flex; justify-content: space-between; align-items: baseline; gap: 1rem;
                 padding: 0.55rem 0; border-bottom: 1px solid {HAIRLINE}; }}
.sq-instr .lab {{ color: {MUTED}; }}
.sq-instr .read {{ text-align: right; }}
.sq-instr .val {{ font-family: {CONDENSED}; font-weight: 600; font-size: 1.08rem; color: {TEXT};
                 font-variant-numeric: tabular-nums; }}
.sq-instr .sub {{ display: block; color: {MUTED}; font-size: 0.78rem; }}

.sq-block {{ margin: 0 0 1.25rem; }}
.sq-block .ttl {{ font-family: {CONDENSED}; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.4rem; }}
.sq-block ul {{ margin: 0; padding-left: 1.15rem; font-size: 0.92rem; line-height: 1.55; }}
.sq-block li::marker {{ color: {DIM}; }}
.sq-block .state {{ font-weight: 600; font-size: 0.95rem; }}
.sq-block .expl {{ color: {MUTED}; font-size: 0.86rem; }}
.sq-line {{ border-left: 3px solid var(--state, {RECT}); padding: 0.15rem 0 0.15rem 0.75rem; font-size: 0.87rem;
           line-height: 1.45; color: var(--state, {TEXT}); margin: 0 0 1.2rem; }}

/* plain-language reasons next to the probability chart */
.sq-reasons .ttl {{ font-family: {CONDENSED}; font-weight: 600; font-size: 16px; color: {TEXT};
                   margin: 0.35rem 0 0.8rem; }}
.sq-reasons .r {{ margin: 0 0 0.85rem; font-size: 0.86rem; line-height: 1.45; }}
.sq-reasons .r b {{ display: block; font-weight: 500; color: {TEXT}; }}
.sq-reasons .r span {{ color: {MUTED}; }}

/* contextual notes (false negative of the 11% rule, empty state) */
.sq-note {{ border-left: 3px solid var(--state, {RECT}); background: rgba(30, 39, 64, 0.55);
           padding: 0.7rem 1rem; border-radius: 0 6px 6px 0; margin: 0.2rem 0 1rem; max-width: 92ch; }}
.sq-empty {{ border: 1px dashed {HAIRLINE}; border-radius: 6px; padding: 1.4rem 1.6rem; max-width: 760px; }}
.sq-empty h3 {{ margin: 0 0 0.6rem; }}
.sq-empty ol {{ margin: 0; padding-left: 1.2rem; color: {MUTED}; line-height: 1.65; }}
.sq-empty ol b {{ color: {TEXT}; font-weight: 500; }}

.sq-section {{ font-family: {CONDENSED}; font-weight: 600; font-size: 1.25rem; margin: 1.6rem 0 0.4rem; }}
.sq-sub {{ color: {MUTED}; font-size: 0.9rem; margin: -0.2rem 0 0.6rem; max-width: 80ch; }}

section[data-testid="stSidebar"] h2 {{ font-size: 1.3rem; }}
section[data-testid="stSidebar"] .sq-group {{ color: {MUTED}; font-size: 0.85rem; margin: 1.1rem 0 0.2rem;
                                              padding-top: 0.8rem; border-top: 1px solid {HAIRLINE}; }}

@media (max-width: 640px) {{
  .block-container {{ padding-top: 3.6rem; }}
  .sq-head .mark {{ font-size: 1.85rem; }}
  .sq-empty {{ padding: 1rem 1.1rem; }}
  .sq-verdict .body {{ padding: 0.9rem 1rem 1rem; }}
  .sq-strip .cell {{ padding-left: 0.7rem; }}
}}

:focus-visible {{ outline: 2px solid {RECT} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def _register_plotly_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=FONT, color=MUTED, size=12),
        title=dict(font=dict(family=CONDENSED, color=TEXT, size=16), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[RECT, DIAG, "#5FD3C6", "#F2A65A", "#E879A6", "#A3D977", "#F5D76E", MUTED],
        xaxis=dict(gridcolor=HAIRLINE, linecolor=HAIRLINE, zerolinecolor=HAIRLINE, tickcolor=HAIRLINE),
        yaxis=dict(gridcolor=HAIRLINE, linecolor=HAIRLINE, zerolinecolor=HAIRLINE, tickcolor=HAIRLINE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=HAIRLINE, font=dict(family=FONT, color=TEXT)),
        margin=dict(l=10, r=10, t=44, b=10),
    )
    pio.templates["shieldqkd"] = tpl
    pio.templates.default = "shieldqkd"


def apply(st) -> None:
    """Inject the stylesheet and make the chart template the default."""
    _register_plotly_template()
    st.markdown(CSS, unsafe_allow_html=True)


_register_plotly_template()          # also when components are imported on their own
