"""Mission-control pieces for the live session: stage tracker, verdict readout,
instrument list, next steps, key status and contextual notes."""
from __future__ import annotations

from html import escape

from dashboard import theme as T

HEADLINE = {
    "PROCEED": "Channel is clean. The key is accepted.",
    "CAUTION": "Probably clean, but not certain enough. The key is withheld.",
    "WARN": "Possible attack with low confidence. The key is destroyed.",
    "ABORT_ATTACK": "Eavesdropping detected on the quantum channel.",
    "ABORT_MITM": "The classical messages were tampered with.",
}

STAGES = ("Handshake", "Transmission", "Sifting", "Analysis", "Verdict", "Use")


def session_stages(out: dict, key_destroyed: bool) -> list[tuple[str, str]]:
    """State of the six real pipeline stages for one finished session.
    States: done, stopped, skipped, final (verdict reached), ready, blocked."""
    ev = out["evaluation"]
    pqc_ok = out["pqc"]["pqc_verified"]
    analysed = bool(ev.get("qber_stats"))
    if out.get("key_id") and not key_destroyed:
        use = "ready"
    elif out.get("key_id"):
        use = "done"
    else:
        use = "blocked"
    return [("Handshake", "done"), ("Transmission", "done"),
            ("Sifting", "done" if pqc_ok else "stopped"),
            ("Analysis", "done" if analysed else "skipped"),
            ("Verdict", "final"), ("Use", use)]


def render_stages(st, stages: list[tuple[str, str]]) -> None:
    tip = {"done": "completed", "stopped": "stopped here", "skipped": "skipped", "final": "reached",
           "ready": "key ready to use", "blocked": "not possible: no key"}
    parts = []
    for i, (name, state) in enumerate(stages):
        if i:
            parts.append('<span class="link" aria-hidden="true"></span>')
        parts.append(f'<span class="st {state}" title="{tip.get(state, state)}">{i + 1}. {escape(name)}</span>')
    st.markdown(f'<nav class="sq-stages" aria-label="Session stages">{"".join(parts)}</nav>',
                unsafe_allow_html=True)


def render_verdict(st, ev: dict, key_status: str | None = None, compact: bool = True) -> None:
    decision = ev["decision"]
    color = T.DECISION_COLOR.get(decision, T.MUTED)
    st.markdown(
        f"""<div class="sq-verdict{' compact' if compact else ''}" style="--state:{color}" role="status" aria-live="polite">
  <div class="beam"></div>
  <div class="body">
    <div class="kicker">Circuit breaker verdict</div>
    <div class="word">{escape(decision)}</div>
    <div class="why">{HEADLINE.get(decision, '')}</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_instruments(st, rows: list[tuple[str, str, str | None, str | None]]) -> None:
    """rows: (label, reading, sub-reading, reading colour)."""
    html = "".join(
        f'<div class="row"><span class="lab">{escape(lab)}</span><span class="read">'
        f'<span class="val"{f" style=color:{col}" if col else ""}>{escape(val)}</span>'
        f'{f"<span class=sub>{escape(sub)}</span>" if sub else ""}</span></div>'
        for lab, val, sub, col in rows
    )
    st.markdown(f'<div class="sq-instr">{html}</div>', unsafe_allow_html=True)


def render_steps(st, title: str, steps: list[str]) -> None:
    items = "".join(f"<li>{escape(s)}</li>" for s in steps)
    st.markdown(f'<div class="sq-block"><div class="ttl">{escape(title)}</div><ul>{items}</ul></div>',
                unsafe_allow_html=True)


def render_key_status(st, state: str, color: str, explanation: str) -> None:
    st.markdown(f'<div class="sq-block"><div class="ttl">Key status</div>'
                f'<div class="state" style="color:{color}">{escape(state)}</div>'
                f'<div class="expl">{escape(explanation)}</div></div>', unsafe_allow_html=True)


def render_line(st, html_text: str, color: str = T.RECT) -> None:
    """Compact coloured note with a left rule."""
    st.markdown(f'<div class="sq-line" style="--state:{color}">{html_text}</div>', unsafe_allow_html=True)


def render_note(st, html_text: str, color: str = T.RECT) -> None:
    st.markdown(f'<div class="sq-note" style="--state:{color}">{html_text}</div>',
                unsafe_allow_html=True)


def render_status_strip(st, cells: list[tuple[str, str, str | None, str | None]]) -> None:
    """Horizontal variant of the instrument list (kept for reuse)."""
    html = "".join(
        f'<div class="cell"><div class="lab">{escape(lab)}</div>'
        f'<div class="val"{f" style=color:{col}" if col else ""}>{escape(val)}</div>'
        f'<div class="note">{escape(note) if note else "&nbsp;"}</div></div>'
        for lab, val, note, col in cells
    )
    st.markdown(f'<div class="sq-strip">{html}</div>', unsafe_allow_html=True)


# kept for backwards compatibility with older imports
def render_alert(st, ev: dict, key_status: str | None = None) -> None:
    render_verdict(st, ev, key_status, compact=False)
