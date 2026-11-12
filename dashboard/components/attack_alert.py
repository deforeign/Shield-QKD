"""Circuit-breaker decision card."""
from __future__ import annotations

STYLE = {
    "PROCEED": ("#1b7f3b", "#e6f4ea", "GREEN - PROCEED", "Key accepted"),
    "CAUTION": ("#9a6b00", "#fff6dc", "AMBER - CAUTION", "Normal but low confidence - key withheld"),
    "WARN": ("#9a6b00", "#fff6dc", "AMBER - WARN", "Possible attack, low confidence"),
    "ABORT_ATTACK": ("#b3261e", "#fde7e7", "RED - ABORT_ATTACK", "Quantum-channel attack detected"),
    "ABORT_MITM": ("#b3261e", "#fde7e7", "RED - ABORT_MITM", "Classical channel tampering detected"),
}


def render_alert(st, ev: dict, key_status: str | None = None):
    fg, bg, head, sub = STYLE.get(ev["decision"], ("#333", "#eee", ev["decision"], ""))
    attack = ev.get("attack_class") or "None"
    conf = ev.get("confidence", 0.0) * 100
    pqc = "VERIFIED" if ev.get("pqc_verified") else "FAILED"
    extra = ""
    if ev.get("resampling_rounds"):
        extra = f"<br/><small>uncertain verdict -> {ev['resampling_rounds']} extra session(s) collected and re-classified</small>"
    st.markdown(
        f"""
<div style="border-left:8px solid {fg};background:{bg};padding:14px 18px;border-radius:8px;color:#111">
  <div style="font-size:1.35rem;font-weight:700;color:{fg}">{head}</div>
  <div style="margin-top:4px">{sub}</div>
  <div style="margin-top:8px;font-size:0.95rem">
    <b>Attack class:</b> {attack} &nbsp;|&nbsp; <b>Confidence:</b> {conf:.1f}% &nbsp;|&nbsp;
    <b>PQC signature:</b> {pqc}<br/>
    <b>Response:</b> {ev.get('recommended_action', '')}<br/>
    <b>Key material:</b> {key_status or '-'}{extra}
  </div>
</div>""",
        unsafe_allow_html=True,
    )
