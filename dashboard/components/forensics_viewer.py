"""Browse generated forensics reports."""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go

from dashboard import theme as T
from dashboard.components.qber_live_chart import qber_figure
from dashboard.components.shap_waterfall import probability_figure
from security.forensics import list_reports


def _summary_rows(paths):
    rows = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        s = d.get("session") or {}
        rows.append({"report_id": d["report_id"], "time (UTC)": d["timestamp_utc"][:19],
                     "decision": d["decision"], "attack_class": d.get("attack_class") or "-",
                     "confidence": round(d.get("confidence", 0) * 100, 1),
                     "true attack": s.get("attack_type") or "none",
                     "qber %": round(100 * (s.get("overall_qber") or 0), 2), "_path": p})
    return rows


def render_forensics(st):
    paths = list_reports()
    if not paths:
        st.markdown('<div class="sq-empty"><h3>No reports yet</h3>Every session writes a report here. '
                    'Run one from the sidebar to create the first.</div>', unsafe_allow_html=True)
        return
    rows = _summary_rows(paths)
    df = pd.DataFrame(rows)
    c1, c2 = st.columns([1, 1])
    decisions = c1.multiselect("Show decisions", sorted(df["decision"].unique()),
                               default=sorted(df["decision"].unique()))
    only_attacks = c2.checkbox("Only sessions where an attack was found", value=False)
    view = df[df["decision"].isin(decisions)]
    if only_attacks:
        view = view[view["decision"].isin(["ABORT_ATTACK", "WARN", "ABORT_MITM"])]
    shown = view.drop(columns=["_path"]).rename(columns={
        "report_id": "Report", "time (UTC)": "Time (UTC)", "decision": "Decision",
        "attack_class": "Attack found", "confidence": "Confidence %", "true attack": "Attack simulated",
        "qber %": "Error rate %"})
    st.dataframe(shown, use_container_width=True, hide_index=True)
    if view.empty:
        return
    choice = st.selectbox("Open report", view["report_id"].tolist())
    path = view.loc[view["report_id"] == choice, "_path"].iloc[0]
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    left, right = st.columns([1.1, 1])
    with left:
        st.code(d.get("text", ""), language="text")
        st.download_button("Download .txt", d.get("text", ""), file_name=f"{choice}.txt")
        pdf = path.replace(".json", ".pdf")
        if os.path.exists(pdf):
            with open(pdf, "rb") as fh:
                st.download_button("Download PDF", fh.read(), file_name=os.path.basename(pdf))
    with right:
        if isinstance(d.get("qber_sequence"), list):
            st.plotly_chart(qber_figure(d["qber_sequence"], height=300), use_container_width=True)
        if d.get("all_probabilities"):
            st.plotly_chart(probability_figure(d["all_probabilities"], height=300),
                            use_container_width=True)
        sh = d.get("shap") or {}
        if sh.get("top_features"):
            tf = sh["top_features"][::-1]
            fig = go.Figure(go.Bar(x=[t["shap_value"] for t in tf], y=[t["feature"] for t in tf],
                                   orientation="h",
                                   marker_color=[T.TEXT if t["shap_value"] > 0 else T.DIM for t in tf]))
            fig.update_layout(title=f"Why the model chose {sh.get('explained_class')}", height=280)
            st.plotly_chart(fig, use_container_width=True)
